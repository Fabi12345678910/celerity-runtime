# GPU 3D Copy Lib [![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/celerity/celerity-runtime/blob/master/LICENSE)

This library intends to provide a simple very high performance way to copy strided data with a 3D layout between GPU devices, and between CPU and GPU memory spaces.

It is implemented in C++ and SYCL, and provides a C++ API.

This library is based on `https://github.com/PeterTh/gpu2dcopylib`, with the main difference being the implementation of a 3rd copy dimension.

This library was specifically designed for the use within celerity, and therefore may have interface features specific to the requirements presented by the celerity runtime.

## Status

The library is implemented: planning (normalization, chunking, staging and the device-to-device routes) and execution
(an executor with a staging slice per worker thread, copy kernels, and an asynchronous `copy_handle`). Every test
passes on SimSYCL, which CI uses, and on AdaptiveCpp's CPU backend, the only asynchronous SYCL implementation it has
run on so far.

Not done yet: benchmarks, a strategy selection based on them, and the integration into Celerity. See
[docs/design.md](docs/design.md) for the design and its open decisions, and [docs/testing.md](docs/testing.md) for the
test layers.

## Prerequisites

- A SYCL implementation
    - The library is intended to work with SimSYCL (in CI), DPC++, and AdaptiveCpp
- CMake 3.24 or later

## Building

This library is meant to be vendored into a larger project:

```cmake
add_subdirectory(vendor/gpu3dcopylib)
target_link_libraries(my_target PRIVATE copylib::copylib)
```

Because the public headers expose SYCL types, the consuming target needs SYCL applied to it as well.
With SimSYCL this happens automatically; with AdaptiveCpp and DPC++ the consumer has to call
`add_sycl_to_target(TARGET my_target)` on its own targets, as device compilation requires it anyway.

It can also be built standalone:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/path/to/sycl
cmake --build build
```

The CMake script will report which SYCL implementation it has found and is using.

### Configuration Options

- `COPYLIB_USE_FMT`: Use the fmt library instead of relying on the C++20 `std::format`. Default: `OFF`
- `COPYLIB_BUILD_TESTS`: Build the test suite. Default: `ON` for top-level builds, `OFF` when vendored

`fmt` has to be findable by CMake if it is enabled; nothing is fetched automatically. Catch2 is fetched at configure time when the tests are built.

### Environment Variables

- `COPYLIB_ALLOC_CPU_IDS`: one CPU ID per device, comma-separated (e.g. `0,16,32,48`). While allocating the host staging buffer of device *i*, the executor runs on CPU *i*, so that the buffer lands on that CPU's NUMA node. Unset by default, in which case host staging is not pinned.
- `COPYLIB_WG_SIZE`: the work-group size of the copy kernels, a positive integer. Default: 128 on Intel GPUs, 32 otherwise.

## Data Layout

All copies are described by a `data_layout`, a box of data inside a larger 3D allocation. `d0` is the innermost dimension and always contiguous, `d2` the outermost; `d0` counts bytes, `d1` rows and `d2` planes:

| Field | Meaning | Unit |
| --- | --- | --- |
| `base` | base address of the allocation (or a `staging_id` for a staging buffer which has not been placed yet) | |
| `d0_stride` | one full row of the allocation | bytes |
| `d1_stride` | one full plane of the allocation | rows |
| `d0_start_offset`, `d0_end_offset` | start and end of the box within a row | bytes |
| `d1_start_offset`, `d1_end_offset` | start and end of the box within a plane | rows |
| `d2_start_offset`, `d2_end_offset` | start and end of the box within the allocation | planes |
| `start`, `end` | half-open window of byte offsets into the packed box, i.e. with the gaps excluded | bytes |

The outermost extent of the allocation is never needed. For an `int` allocation of shape `[12, 16, 20]`, `d0_stride` is `20 * 4` and `d1_stride` is `16`.

The window selects which part of the box a copy actually transfers. The constructors set it to cover the whole box;
chunking keeps the box and narrows only the window, so a chunk is described by the same nine box values as the copy it
came from. A copy spec is valid when both of its windows have the same length, which is what allows a reshaping copy
such as `int[1,1,6]` to `int[1,2,3]`.

## Usage

To use the library for copy operations, first, an instance of an executor must be created.  
Then, copy operations can be specified, turned into optimized parallel copy sets, and executed.

Here is a complete example which manually specifies all parameters to serve as a reference:

```cpp
#include <copylib/copylib.hpp>

using namespace copylib;

// === 1. Initialization

const int64_t buffer_size = 128 * 1024 * 1024; // 128 MiB for staging buffers
const int64_t queues_per_device = 2; // number of in-order queues per device for asynchronicity
executor exec(buffer_size, 2, queues_per_device); // create an executor
utils::print(exec.get_info()); // [optional] print information about the execution environment

// === 2. Specifying a copy operation

// provide these pointers as appropriate to source and target memory
intptr_t src_ptr = 0x1000; // pointer to the source (must be at least 2-byte aligned)
intptr_t dst_ptr = 0x2000; // pointer to the destination

// an int allocation of shape [12, 16, 20], of which the box [2, 5) x [3, 7) x [4, 10) is copied
const int64_t elem = sizeof(int);
const int64_t d0_stride = 20 * elem; // bytes per row
const int64_t d1_stride = 16; // rows per plane

// source data layout: d0 in bytes, d1 in rows, d2 in planes
const data_layout source_layout{src_ptr, d0_stride, d1_stride, 4 * elem, 3, 2, 10 * elem, 7, 5};
const data_layout target_layout{dst_ptr, source_layout}; // target data layout, same structure as the source

// copy from device 0 to device 1
const copy_spec spec{device_id::d0, source_layout, device_id::d1, target_layout};
COPYLIB_ENSURE(is_valid(spec), "Invalid copy spec: {}", spec); // [optional] check if the copy spec is valid

// === 3. Manifesting the copy operations into an optimized parallel copy set

const copy_type type = copy_type::staged; // perform linearization
const copy_properties props = copy_properties::use_kernel; // use a kernel for linearization, generally faster
const d2d_implementation d2d = d2d_implementation::host_staging_at_source; // use manual host staging
const int64_t chunk_size = 1024*1024; // generate 1 MiB chunks
const copy_strategy strat(type, props, d2d, chunk_size); // create a strategy
const auto copy_set = manifest_strategy(spec, strat, basic_staging_provider{}); // manifest the copy set
COPYLIB_ENSURE(is_valid(copy_set), "Invalid copy set: {}", copy_set); // [optional] validate the copy set

// === 4. Executing the copy set

const auto handle = execute_copy(exec, copy_set); // returns immediately
handle.wait(); // blocks until every plan has finished
COPYLIB_ENSURE(!handle.error(), "Copy failed: {}", *handle.error());
```

## Benchmarks and Utilities

Not yet ported from the 2D library:

- `utils/info`: Print information about the execution environment and its features
- `benchmarks/manifest`: Micro-benchmark measuring strategy manifesting performance
- `benchmarks/intra_device`: Benchmark for intra-device linearization performance
- `benchmarks/chunk_parallel`: Benchmark for optimized device-to-device copy performance
- `benchmarks/full_set`: Perform a very large run of various benchmarks to characterize platform performance
