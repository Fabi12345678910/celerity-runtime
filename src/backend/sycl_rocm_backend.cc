#include "async_event.h"
#include "backend/sycl_backend.h"
#include "gpu2dcopylib/lib/copylib_backend.hpp"
#include "gpu2dcopylib/lib/copylib_core.hpp"
#include "nd_memory.h"
#include "utils.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <iostream>
#include <gpu2dcopylib/lib/copylib.hpp>

namespace celerity::detail {
namespace sycl_backend_detail{
using namespace copylib;

void nd_copy_rocm_device_linear(sycl::queue& queue, const void* const source, void* const dest, size_t size_bytes,const size_t elem_size,
    bool enable_profiling, std::optional<sycl::event>& first, sycl::event& last){

    const int64_t buffer_size = static_cast<const int64_t>(128 * 1024) * 1024; // 128 MiB for staging buffers
    const int64_t queues_per_device = 1; // number of in-order queues per device for asynchronicity
    executor exec(buffer_size,1, queues_per_device); // create an executor

    intptr_t source_ptr = reinterpret_cast<intptr_t>(source);
    intptr_t dest_ptr = reinterpret_cast<intptr_t>(dest);

    const data_layout source_layout{source_ptr, 0, static_cast<int64_t>(elem_size), 1, 0}; // source data layout
    const data_layout target_layout{dest_ptr, source_layout}; // target data layout, same structure as the source

    const copy_spec spec{copylib::device_id::host, source_layout, copylib::device_id::d0, target_layout};
    COPYLIB_ENSURE(is_valid(spec), "Invalid copy spec: {}", spec); // [optional] check if the copy spec is valid

    
    const copy_type type = copy_type::staged; // perform linearization
    const copy_properties props = copy_properties::use_kernel; // use a kernel for linearization, generally faster
    const d2d_implementation d2d = d2d_implementation::host_staging_at_source; // use manual host staging
    const int64_t chunk_size = static_cast<const int64_t>(1024*1024); // generate 1 MiB chunks
    const copylib::copy_strategy strat(type, props, d2d, chunk_size); // create a strategy
    const auto copy_set = manifest_strategy(spec, strat, basic_staging_provider{}); // manifest the copy set

    COPYLIB_ENSURE(is_valid(copy_set), "Invalid copy set: {}", copy_set); // [optional] validate the copy set
    copylib::execute_copy(exec, copy_set);


//        last = queue.memcpy(dest, source, size_bytes);
    if(enable_profiling) { first = last; }
    }

void nd_copy_rocm_device_chunked(sycl::queue& queue, const void* const source_base, void* const dest_base, const box<3>& source_box, const box<3>& dest_box,
    const box<3>& copy_box, const size_t elem_size, bool enable_profiling, std::optional<sycl::event>& first, sycl::event& last){

    const auto layout = layout_nd_copy(source_box.get_range(), dest_box.get_range(), copy_box.get_offset() - source_box.get_offset(),
    copy_box.get_offset() - dest_box.get_offset(), copy_box.get_range(), elem_size);

	// nd_copy_layout is defined such that we can ignore num_complex_strides in this loop and will perform exactly one iteration for every non-complex stride.
	size_t source_offset_0 = layout.offset_in_source;
	size_t dest_offset_0 = layout.offset_in_dest;
    if(layout.strides[1] != nd_copy_layout::stride_dimension()){
            utils::panic("multi strided copy is not available as of yet");
        }
    

    const int64_t buffer_size = static_cast<const int64_t>(128 * 1024) * 1024; // 128 MiB for staging buffers
    const int64_t queues_per_device = 1; // number of in-order queues per device for asynchronicity
    executor exec(buffer_size, 1, queues_per_device); // create an executor

    intptr_t source_ptr = reinterpret_cast<intptr_t>(source_base);
    intptr_t dest_ptr = reinterpret_cast<intptr_t>(dest_base);

    const data_layout source_layout{source_ptr,
        static_cast<int64_t>(source_offset_0),
        static_cast<int64_t>(layout.contiguous_size),
        static_cast<int64_t>(layout.strides[0].count),
        static_cast<int64_t>(layout.strides[0].source_stride)};

    const data_layout target_layout{dest_ptr,
        static_cast<int64_t>(dest_offset_0),
        static_cast<int64_t>(layout.contiguous_size),
        static_cast<int64_t>(layout.strides[0].count),
        static_cast<int64_t>(layout.strides[0].dest_stride)};

    const copy_spec spec{copylib::device_id::host, source_layout, copylib::device_id::d0, target_layout};
    COPYLIB_ENSURE(is_valid(spec), "Invalid copy spec: {}", spec); // [optional] check if the copy spec is valid

    
    const copy_type type = copy_type::staged; // perform linearization
    const copy_properties props = copy_properties::use_kernel; // use a kernel for linearization, generally faster
    const d2d_implementation d2d = d2d_implementation::host_staging_at_source; // use manual host staging
    const int64_t chunk_size = static_cast<const int64_t>(1024*1024); // generate 1 MiB chunks
    const copylib::copy_strategy strat(type, props, d2d, chunk_size); // create a strategy
    const auto copy_set = manifest_strategy(spec, strat, basic_staging_provider{}); // manifest the copy set

    COPYLIB_ENSURE(is_valid(copy_set), "Invalid copy set: {}", copy_set); // [optional] validate the copy set
    auto event = copylib::execute_copy(exec, copy_set);
    if(!event.has_value()){
        utils::panic("copylib did not create an sycl::event");
    }
    last = event.value();
//	for(size_t i = 0; i < layout.strides[0].count; ++i) {
//		size_t source_offset_1 = source_offset_0;
//		size_t dest_offset_1 = dest_offset_0;
//        if(layout.strides[1].count != 1){
//            utils::panic("multi strided copy is not available as of yet");
//        }
//		for(size_t j = 0; j < layout.strides[1].count; ++j) {
//			f(source_offset_1, dest_offset_1, layout.contiguous_size);
//			source_offset_1 += layout.strides[1].source_stride;
//			dest_offset_1 += layout.strides[1].dest_stride;
//		}
//		source_offset_0 += layout.strides[0].source_stride;
//		dest_offset_0 += layout.strides[0].dest_stride;
//	}
    if(enable_profiling) { first = last; }

}

async_event nd_copy_device_rocm(sycl::queue& queue, const void* const source_base, void* const dest_base, const region_layout& source_layout,
    const region_layout& dest_layout, const region<3>& copy_region, const size_t elem_size, bool enable_profiling) //
{
	// We remember the first and last submission event to report completion time spanning the entire region copy.
	std::optional<sycl::event> first; // set only if profiling is enabled
	sycl::event last;
	dispatch_nd_region_copy(
	    source_base, dest_base, source_layout, dest_layout, copy_region, elem_size,
	    [&queue, elem_size, enable_profiling, &first, &last](const void* const source, void* const dest, const box<3>& source_box, const box<3>& dest_box,
	        const box<3>& copy_box) { nd_copy_rocm_device_chunked(queue, source, dest, source_box, dest_box, copy_box, elem_size, enable_profiling, first, last); },
	    [&queue, elem_size, enable_profiling, &first, &last](const void* const source, void* const dest, size_t size_bytes) {
		    nd_copy_rocm_device_linear(queue, source, dest, size_bytes, elem_size, enable_profiling, first, last);
	    });

	flush(queue);
	return make_async_event<sycl_event>(std::move(first), std::move(last));
}

}

sycl_rocm_backend::sycl_rocm_backend(const std::vector<sycl::device>& devices, const sycl_backend::configuration& config) : sycl_backend(devices, config) {
	// CUDA permits cudaMemcpy between devices that are not peer-enabled, but will implicitly stage the copy through host memory, which wreaks havoc on stream
	// parallelism (see https://forums.developer.nvidia.com/t/queueing-device-to-device-peer-memcpy-stalls-concurrent-copy-operations/295894). We therefore
	// choose not to consider such GPUs to be copy-peers. There is potential to improve performance by partially overlapping the corresponding D2H and H2D
	// copies, but this must be expressible in the IDAG (TODO).
	for(device_id i = 0; i < devices.size(); ++i) {
		for(device_id j = i + 1; j < devices.size(); ++j) {
            //perhaps enable device2device peer copy at this point
		}
	}
}

async_event sycl_rocm_backend::enqueue_device_copy(device_id device, size_t device_lane, const void* const source_base, void* const dest_base,
			const region_layout& source_layout, const region_layout& dest_layout, const region<3>& copy_region, const size_t elem_size)
    { 
        return enqueue_device_work(device, device_lane, [=, this](sycl::queue& queue) {
            return sycl_backend_detail::nd_copy_device_rocm(
                queue, source_base, dest_base, source_layout, dest_layout, copy_region, elem_size, is_profiling_enabled());
        });
    }
}