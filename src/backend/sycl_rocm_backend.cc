#include "async_event.h"
#include "backend/sycl_backend.h"
#include <hip/hip_runtime.h>

namespace celerity::detail {
namespace sycl_backend_detail{

async_event nd_copy_device_rocm(sycl::queue& queue, const void* const source_base, void* const dest_base, const region_layout& source_layout,
    const region_layout& dest_layout, const region<3>& copy_region, const size_t elem_size, bool enable_profiling) //
{
    //for now only support one block of memory
    std::cout << "doing our custom memcpy!\n";
    
    return async_event();
    //hipMemcpy(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind)
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