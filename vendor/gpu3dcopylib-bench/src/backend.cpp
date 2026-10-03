#include <copylib/backend.hpp>

#include <copylib/support.hpp> // IWYU pragma: keep - this is needed for formatting output, IWYU is dumb

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

#include <pthread.h>

namespace copylib {

std::string executor::get_sycl_impl_name() const {
#if defined(SIMSYCL_VERSION)
	return "SimSYCL";
#elif defined(__ADAPTIVECPP__)
	std::string ret = "AdaptiveCpp";
#if defined(__ACPP_ENABLE_CUDA_TARGET__)
	ret += " (CUDA)";
#endif // __ACPP_ENABLE_CUDA_TARGET__
	return ret;
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
	return "IntelSYCL";
#else
	return "Unknown SYCL implementation";
#endif
}

bool executor::is_device_to_device_copy_available() const {
#if defined(SIMSYCL_VERSION)
	return true;
#elif defined(__ADAPTIVECPP__)
#if defined(__ACPP_ENABLE_CUDA_TARGET__)
	return true; // CUDA emulates p2p transfers even if not available
#else
	return false; // assumption for now
#endif
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
	if(devices.empty()) { return false; }
	return devices.front().dev.get_info<sycl::info::device::vendor>().find("NVIDIA") != std::string::npos;
#else
	return false;
#endif
}

bool executor::is_peer_memory_access_available() const { return peer_access_available; }

int32_t executor::get_preferred_wg_size() const { return preferred_wg_size; }

std::optional<int> get_cpu_for_gpu_alloc(int gpu_idx, size_t total_gpu_count) {
	constexpr int max_gpu_idx = static_cast<int>(device_id::count);
	COPYLIB_ENSURE(gpu_idx >= 0 && static_cast<size_t>(gpu_idx) < total_gpu_count, "Invalid gpu index: {} (needs to be >=0 and <{})", gpu_idx, total_gpu_count);
	COPYLIB_ENSURE(total_gpu_count <= max_gpu_idx, "Invalid total gpu count: {} (needs to be <={})", total_gpu_count, max_gpu_idx);
	const auto env_var = std::getenv("COPYLIB_ALLOC_CPU_IDS");
	if(env_var == nullptr) { return std::nullopt; }
	const auto cpu_ids_split = utils::split(std::string(env_var), ',');
	COPYLIB_ENSURE(cpu_ids_split.size() >= total_gpu_count, "Insufficient number of CPU IDs provided in COPYLIB_ALLOC_CPU_IDS: {} (expected {})",
	    cpu_ids_split.size(), total_gpu_count);
	const auto& cpu_id = cpu_ids_split[gpu_idx];
	int cpu = -1;
	const auto [end, error] = std::from_chars(cpu_id.data(), cpu_id.data() + cpu_id.size(), cpu);
	COPYLIB_ENSURE(
	    error == std::errc{} && end == cpu_id.data() + cpu_id.size() && cpu >= 0 && cpu < CPU_SETSIZE, "Invalid CPU ID in COPYLIB_ALLOC_CPU_IDS: '{}'", cpu_id);
	return cpu;
}

std::string executor::get_info() const {
	auto ret = utils::format("Copylib executor with {} device(s) and buffer size {} bytes\n", devices.size(), buffer_size);
	ret += utils::format("SYCL implementation: {}\n", get_sycl_impl_name());
	ret += utils::format("D2D copy: {}    Peer access: {}    Preferred wg size: {}\n", //
	    is_device_to_device_copy_available(), is_peer_memory_access_available(), get_preferred_wg_size());
	ret += utils::format("Using {} queues per device, with {} bytes of staging per worker\n", get_queues_per_device(), get_staging_slice_size());
	for(size_t i = 0; i < devices.size(); i++) {
		ret += utils::format("    Device {:2}: {} [{}]", i, //
		    devices[i].dev.get_info<sycl::info::device::name>(), devices[i].dev.get_info<sycl::info::device::vendor>());
		const auto& cpu = devices[i].host_staging_cpu;
		ret += cpu ? utils::format(" (host alloc on core {})\n", *cpu) : std::string(" (host alloc not pinned)\n");
	}
	return ret;
}

executor::possibility executor::can_copy(const copy_spec& spec) const {
	const bool d2d = is_device_to_device_copy_available();
	const auto d2d_copy = spec.source_device != spec.target_device && spec.source_device != device_id::host && spec.target_device != device_id::host;
	if(d2d_copy) {
		if(spec.properties & copy_properties::use_kernel) { return possibility::needs_d2d_copy; } // TODO need to be more specific here later
		if(!d2d) { return possibility::needs_d2d_copy; }
	}
	return possibility::possible;
}

executor::possibility executor::can_copy(const parallel_copy_set& cset) const {
	for(const auto& plan : cset) {
		for(const auto& spec : plan) {
			const auto res = can_copy(spec);
			if(res != possibility::possible) { return res; }
		}
	}
	return possibility::possible;
}

void executor::barrier() {
	for(auto& dev : devices) {
		for(auto& q : dev.queues) {
			q.wait_and_throw();
		}
	}
}

namespace {

	// checked before the pool is built from it, which would start a thread per hardware thread for 0
	std::size_t checked_queues_per_device(int64_t queues_per_device) {
		COPYLIB_ENSURE(queues_per_device > 0, "Need at least one queue per device");
		return static_cast<std::size_t>(queues_per_device);
	}

	// enables peer access between every pair of devices, and reports whether every pair supports it
	bool enable_peer_access([[maybe_unused]] detail::device_list& devices) {
#if defined(SIMSYCL_VERSION)
		return true;
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
		for(size_t dev_idx_a = 0; dev_idx_a < devices.size(); dev_idx_a++) {
			for(size_t dev_idx_b = 0; dev_idx_b < devices.size(); dev_idx_b++) {
				if(dev_idx_a == dev_idx_b) { continue; }
				if(devices[dev_idx_a].dev.ext_oneapi_can_access_peer(devices[dev_idx_b].dev)) {
					devices[dev_idx_a].dev.ext_oneapi_enable_peer_access(devices[dev_idx_b].dev);
				} else {
					return false;
				}
			}
		}
		return true;
#else
		return false;
#endif
	}

	// pairs each device with the context a queue created from the device alone gets, i.e. the implementation's default
	// context, which a caller allocating through such queues uses as well
	std::vector<std::pair<sycl::device, sycl::context>> with_default_contexts(const std::vector<sycl::device>& devices) {
		std::vector<std::pair<sycl::device, sycl::context>> device_contexts;
		for(const auto& device : devices) {
			device_contexts.emplace_back(device, sycl::queue(device).get_context());
		}
		return device_contexts;
	}

	std::vector<std::pair<sycl::device, sycl::context>> first_gpus(int64_t devices_needed) {
		COPYLIB_ENSURE(devices_needed > 0, "Need at least one device");
		auto gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
		if(gpu_devices.size() < static_cast<size_t>(devices_needed)) {
			COPYLIB_ERROR("Not enough GPU devices available: {} ({} needed)", gpu_devices.size(), devices_needed);
		} else if(gpu_devices.size() > static_cast<size_t>(devices_needed)) {
			gpu_devices.resize(devices_needed); // don't waste time initializing more devices than needed
		}
		return with_default_contexts(gpu_devices);
	}

	// COPYLIB_WG_SIZE if set, otherwise a default for the vendor of the first device
	int32_t read_preferred_wg_size(const sycl::device& first_device) {
		const auto env_var = std::getenv("COPYLIB_WG_SIZE");
		if(env_var == nullptr) { return first_device.get_info<sycl::info::device::vendor>().find("Intel") != std::string::npos ? 128 : 32; }
		const auto env_end = env_var + std::strlen(env_var);
		int32_t wg_size = 0;
		const auto [end, error] = std::from_chars(env_var, env_end, wg_size);
		COPYLIB_ENSURE(error == std::errc{} && end == env_end && wg_size > 0, "Invalid work-group size in COPYLIB_WG_SIZE: '{}'", env_var);
		return wg_size;
	}

} // namespace

executor::executor(int64_t buffer_size) : executor(buffer_size, sycl::device::get_devices(sycl::info::device_type::gpu).size(), 1) {}

executor::executor(int64_t buffer_size, int64_t devices_needed, int64_t queues_per_device)
    : executor(buffer_size, first_gpus(devices_needed), queues_per_device) {}

executor::executor(int64_t buffer_size, const std::vector<sycl::device>& devices, int64_t queues_per_device)
    : executor(buffer_size, with_default_contexts(devices), queues_per_device) {}

executor::executor(int64_t buffer_size, const std::vector<std::pair<sycl::device, sycl::context>>& device_contexts, int64_t queues_per_device)
    : buffer_size(buffer_size), pool(checked_queues_per_device(queues_per_device), [](const std::size_t idx) {
	      if(std::getenv("COPYLIB_WORKER_CPUS") == nullptr) return;
	      const auto cpus = utils::split(std::getenv("COPYLIB_WORKER_CPUS"), ',');
	      cpu_set_t mask;
	      CPU_ZERO(&mask);
	      if(cpus[0] == "all") {
		      for(unsigned cpu = 0; cpu < std::thread::hardware_concurrency(); cpu++) {
			      CPU_SET(cpu, &mask);
		      }
	      } else {
		      CPU_SET(std::atoi(cpus[idx % cpus.size()].c_str()), &mask);
	      }
	      pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
      }) {
	COPYLIB_ENSURE(!device_contexts.empty(), "Need at least one device");
	COPYLIB_ENSURE(device_contexts.size() <= static_cast<size_t>(device_id::count), "Too many devices: {} (at most {})", device_contexts.size(),
	    static_cast<int>(device_id::count));
	for(const auto& [device, context] : device_contexts) {
		const auto context_devices = context.get_devices();
		COPYLIB_ENSURE(std::find(context_devices.begin(), context_devices.end(), device) != context_devices.end(),
		    "Device {} is not part of the context given with it", device.get_info<sycl::info::device::name>());
	}
	preferred_wg_size = read_preferred_wg_size(device_contexts.front().first);

	// restores the calling thread's affinity however the constructor is left, so a failed check cannot leave it pinned
	struct affinity_guard {
		cpu_set_t prior;
		affinity_guard() {
			CPU_ZERO(&prior);
			COPYLIB_ENSURE(pthread_getaffinity_np(pthread_self(), sizeof(prior), &prior) == 0, "Failed to get CPU affinity");
		}
		~affinity_guard() { pthread_setaffinity_np(pthread_self(), sizeof(prior), &prior); }
	};
	std::optional<affinity_guard> restore_affinity; // only once something is pinned

	COPYLIB_ENSURE(buffer_size > 0, "Invalid buffer size: {} (needs to be > 0)", buffer_size);
	// a whole number of alignments, which aligned allocations require on some implementations; reported as the buffer size
	this->buffer_size = (buffer_size + detail::staging_alignment - 1) / detail::staging_alignment * detail::staging_alignment;
	const auto staging_bytes = static_cast<size_t>(this->buffer_size);
	// one slice per worker, each placing the staging of the plan it runs at the start of its own slice
	staging_slice_size = this->buffer_size / static_cast<int64_t>(pool.get_thread_count()) / detail::staging_alignment * detail::staging_alignment;
	COPYLIB_ENSURE(staging_slice_size > 0, "Buffer size {} leaves less than {} bytes of staging for each of the {} workers", this->buffer_size,
	    detail::staging_alignment, pool.get_thread_count());

	// rethrows asynchronous errors from wait_and_throw, so that they are reported like any other failure; SYCL's default
	// handler would terminate the process instead
	const sycl::async_handler rethrow_async_errors = [](sycl::exception_list errors) {
		for(const auto& e : errors) {
			std::rethrow_exception(e);
		}
	};

	// allocate queues and staging buffers
	int dev_id = 0;
	for(const auto& [device, context] : device_contexts) {
		const sycl::property_list queue_properties = {
		    sycl::property::queue::in_order{},
#ifdef ACPP_EXT_COARSE_GRAINED_EVENTS
		    // minor perf improvement on AdaptiveCpp
		    sycl::property::queue::AdaptiveCpp_coarse_grained_events{},
#endif // ACPP_EXT_COARSE_GRAINED_EVENTS
#ifdef SYCL_EXT_INTEL_QUEUE_IMMEDIATE_COMMAND_LIST
		    // ~ 10% perf improvement on Intel GPUs in strided chunk copy set peak performance
		    sycl::ext::intel::property::queue::immediate_command_list{},
#endif // SYCL_EXT_INTEL_QUEUE_IMMEDIATE_COMMAND_LIST
		};

		std::vector<sycl::queue> queues;
		for(int64_t i = 0; i < queues_per_device; i++) {
			queues.emplace_back(sycl::queue(context, device, rethrow_async_errors, queue_properties));
		}
		// allocated only once the device is listed, so that its destructor frees them if a later check throws
		auto& dev = devices.emplace_back(device, queues);
		auto& q = dev.queues[0];

		dev.staging_buffer = sycl::aligned_alloc_device<std::byte>(detail::staging_alignment, staging_bytes, q);
		COPYLIB_ENSURE(dev.staging_buffer != nullptr, "Failed to allocate device staging buffer");

		// pinned only on request: the calling thread runs on the CPU given for this device while allocating its host staging
		dev.host_staging_cpu = get_cpu_for_gpu_alloc(dev_id, device_contexts.size());
		if(dev.host_staging_cpu) {
			if(!restore_affinity) { restore_affinity.emplace(); }
			cpu_set_t mask_for_device;
			CPU_ZERO(&mask_for_device);
			CPU_SET(*dev.host_staging_cpu, &mask_for_device);
			COPYLIB_ENSURE(pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask_for_device) == 0,
			    "Failed to set CPU affinity to CPU {} for device {}", *dev.host_staging_cpu, dev_id);
		}

		dev.host_staging_buffer = std::getenv("COPYLIB_HOST_STAGING") && !std::strcmp(std::getenv("COPYLIB_HOST_STAGING"), "pageable")
		                              ? static_cast<std::byte*>(std::aligned_alloc(detail::staging_alignment, staging_bytes))
		                              : sycl::aligned_alloc_host<std::byte>(detail::staging_alignment, staging_bytes, q);
		COPYLIB_ENSURE(dev.host_staging_buffer != nullptr, "Failed to allocate host staging buffer");
		std::memset(dev.host_staging_buffer, 0, staging_bytes); // first touch, placing the pages close to the CPU pinned above, if any

		dev_id++;
	}
	peer_access_available = enable_peer_access(devices);

	if(std::getenv("COPYLIB_WARMUP") && std::atoi(std::getenv("COPYLIB_WARMUP"))) {
		for(auto& dev : devices) {
			dev.queues[0].memset(dev.staging_buffer, 0, staging_bytes).wait();
			const auto base = reinterpret_cast<intptr_t>(dev.staging_buffer);
			for(int64_t size = 1; size <= 64; size *= 2) {
				detail::copy_with_kernel(dev.queues[0], copy_spec(device_id::d0, data_layout(base, 0, size), device_id::d0, data_layout(base, 64, size)), preferred_wg_size)
				    .wait();
			}
		}
	}
}

detail::device::device(sycl::device dev, const std::vector<sycl::queue>& queues) : dev(dev), queues(queues) {}

detail::device::~device() {
	for(auto& q : queues) {
		q.wait(); // don't throw in the destructor
	}

	auto& q = queues[0];
	sycl::free(staging_buffer, q);
	if(std::getenv("COPYLIB_HOST_STAGING") && !std::strcmp(std::getenv("COPYLIB_HOST_STAGING"), "pageable")) {
		std::free(host_staging_buffer);
	} else {
		sycl::free(host_staging_buffer, q);
	}
}

sycl::queue& executor::get_queue(device_id id, int64_t queue_idx) {
	COPYLIB_ENSURE(static_cast<int>(id) >= 0 && static_cast<size_t>(id) < devices.size(), "Invalid device id: {} ({} device(s) available)", id, devices.size());
	auto& queues = devices[static_cast<int>(id)].queues;
	COPYLIB_ENSURE(queue_idx >= 0 && static_cast<size_t>(queue_idx) < queues.size(), "Invalid queue idx: {} ({} queue(s) available)", queue_idx, queues.size());
	return queues[queue_idx];
}

sycl::queue& executor::get_queue(const target& tgt) { return get_queue(tgt.did, tgt.queue_idx); }

std::byte* executor::get_staging_buffer(device_id id) {
	COPYLIB_ENSURE(static_cast<int>(id) >= 0 && static_cast<size_t>(id) < devices.size(), "Invalid device id: {} ({} device(s) available)", id, devices.size());
	return devices[static_cast<int>(id)].staging_buffer;
}

std::byte* executor::get_host_staging_buffer(device_id id) {
	COPYLIB_ENSURE(static_cast<int>(id) >= 0 && static_cast<size_t>(id) < devices.size(), "Invalid device id: {} ({} device(s) available)", id, devices.size());
	return devices[static_cast<int>(id)].host_staging_buffer;
}

int64_t executor::get_buffer_size() const { return buffer_size; }

int64_t executor::get_staging_slice_size() const { return staging_slice_size; }

int64_t executor::get_queues_per_device() const { return devices.front().queues.size(); }

namespace detail {

	step_result execute_copy(executor& exec, const copy_spec& spec, int64_t queue_idx, bool alternate_device, step_result last) {
		const executor::target last_target = last.target;
		const device_id last_device = last_target.did;

		const auto& source = spec.source_layout;
		const auto& target_layout = spec.target_layout;

		//  for host <-> host copies, use memcpy
		if(spec.source_device == device_id::host && spec.target_device == device_id::host) {
			if(last_device != device_id::host && last_device != device_id::count) {
				while(std::getenv("COPYLIB_WAIT") && !std::strcmp(std::getenv("COPYLIB_WAIT"), "poll")
				      && last.event.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete) {}
				last.event.wait_and_throw();
			}
			for_each_copy_run(spec, [&](int64_t source_offset, int64_t target_offset, int64_t length) {
				std::memcpy(target_layout.base_ptr() + target_offset, source.base_ptr() + source_offset, length);
			});
			return {{device_id::host, 0}, {}};
		}

		const device_id desired_device = alternate_device ? spec.target_device : spec.source_device;
		const device_id fallback_device = alternate_device ? spec.source_device : spec.target_device;
		const device_id device_to_use = desired_device == device_id::host ? fallback_device : desired_device;
		const executor::target target{device_to_use, queue_idx};

		if(last_target != target && last_device != device_id::count && last_device != device_id::host) {
			while(std::getenv("COPYLIB_WAIT") && !std::strcmp(std::getenv("COPYLIB_WAIT"), "poll")
			      && last.event.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete) {}
			last.event.wait_and_throw();
		}

		auto& queue = exec.get_queue(target);

		// if the source and target are contiguous, we can use a single copy operation
		if(source.is_window_contiguous() && target_layout.is_window_contiguous()) {
			return {target, queue.copy(source.base_ptr() + source.offset_at(source.start),
			                    target_layout.base_ptr() + target_layout.offset_at(target_layout.start), source.window_length())};
		}

		// technically, one could use a kernel for copies involving the host on some hw/sw stacks, but we'll ignore that for now
		if(spec.properties & copy_properties::use_kernel
		    && ((std::getenv("COPYLIB_HOST_KERNEL") && std::atoi(std::getenv("COPYLIB_HOST_KERNEL")))
		        || (spec.source_device != device_id::host && spec.target_device != device_id::host))) {
			return {target, copy_with_kernel(queue, spec, exec.get_preferred_wg_size())};
		}
		// the queue is in order, so the last copy completes after all the others
		sycl::event last_copy;
		for_each_copy_run(spec, [&](int64_t source_offset, int64_t target_offset, int64_t length) {
			last_copy = queue.copy(source.base_ptr() + source_offset, target_layout.base_ptr() + target_offset, length);
		});
		return {target, last_copy};
	}

	staging_fulfiller::staging_fulfiller(executor& exec, int64_t slice) : exec(exec), slice_offset(slice * exec.get_staging_slice_size()) {}

	void staging_fulfiller::fulfill(data_layout& layout) {
		if(!layout.is_unplaced_staging()) { return; }
		const auto staging_idx = layout.staging.index;
		auto staging_it = staging_buffers.find(staging_idx);
		if(staging_it == staging_buffers.end()) {
			const auto did = layout.staging.did;
			const bool host = layout.staging.on_host;
			COPYLIB_ENSURE(did != device_id::host, "Device id for staging cannot be host");
			staging_info info{
			    .size = layout.window_length(),
			    .device = did,
			    .on_host = host,
			};
			if(host) {
				info.buffer = exec.get_host_staging_buffer(did) + slice_offset + current_host_staging_offsets[static_cast<size_t>(did)];
				current_host_staging_offsets[static_cast<size_t>(did)] += (info.size + staging_alignment - 1) / staging_alignment * staging_alignment;
				COPYLIB_ENSURE(current_host_staging_offsets[static_cast<size_t>(did)] <= exec.get_staging_slice_size(),
				    "Staging buffer overflow on host for device {}: a plan needs more than a worker's {} bytes", static_cast<int>(did),
				    exec.get_staging_slice_size());
			} else {
				info.buffer = exec.get_staging_buffer(did) + slice_offset + current_staging_offsets[static_cast<size_t>(did)];
				current_staging_offsets[static_cast<size_t>(did)] += (info.size + staging_alignment - 1) / staging_alignment * staging_alignment;
				COPYLIB_ENSURE(current_staging_offsets[static_cast<size_t>(did)] <= exec.get_staging_slice_size(),
				    "Staging buffer overflow for device {}: a plan needs more than a worker's {} bytes", static_cast<int>(did), exec.get_staging_slice_size());
			}
			staging_it = staging_buffers.emplace(staging_idx, info).first;
		} else {
			COPYLIB_ENSURE(staging_buffers[staging_idx].size == layout.window_length(), "Staging buffer size mismatch");
			COPYLIB_ENSURE(staging_buffers[staging_idx].device == layout.staging.did, "Staging buffer device mismatch");
			COPYLIB_ENSURE(staging_buffers[staging_idx].on_host == layout.staging.on_host, "Staging buffer host flag mismatch");
		}
		layout.base = reinterpret_cast<intptr_t>(staging_it->second.buffer);
	}

	void staging_fulfiller::fulfill(copy_spec& spec) {
		fulfill(spec.source_layout);
		fulfill(spec.target_layout);
	}

} // namespace detail

namespace {

	template <typename T>
	concept StagingFulfiller = requires(T f, copy_spec c) {
		{ f.fulfill(c) };
	};

	void execute_plan_impl(executor& exec, const copy_plan& plan, StagingFulfiller auto& fulfiller, int64_t queue_idx, bool alternate_device) {
		detail::step_result last;
		for(auto spec : plan) {
			fulfiller.fulfill(spec);
			const auto start = std::chrono::steady_clock::now();
			last = detail::execute_copy(exec, spec, queue_idx, alternate_device, last);
			if(std::getenv("COPYLIB_TRACE") && std::atoi(std::getenv("COPYLIB_TRACE"))) {
				last.event.wait_and_throw();
				utils::err_print("copylib trace: {} -> {}, {} bytes in {} ms\n", spec.source_device, spec.target_device, spec.source_layout.window_length(),
				    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
			}
		}
		// the plan is done only once its last step is; earlier steps are covered by the waits between steps and queue order
		while(std::getenv("COPYLIB_WAIT") && !std::strcmp(std::getenv("COPYLIB_WAIT"), "poll")
		      && last.event.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete) {}
		last.event.wait_and_throw();
	}

} // namespace

namespace detail {

	struct copy_state {
		std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
		std::chrono::steady_clock::time_point end;
		std::atomic<int64_t> unfinished_plans = 0;
		std::atomic<bool> complete = false; // set last, so that the end time and the error are in place once it is true
		std::optional<std::string> error;
		std::mutex mutex;
		std::condition_variable completed;

		void fail(const std::string& message) {
			const std::lock_guard lock(mutex);
			if(!error) { error = message; }
		}

		// called once for every plan, successful or not; the last one completes the call
		void finish_plan() {
			if(unfinished_plans.fetch_sub(1) == 1) { finish_call(); }
		}

		void finish_call() {
			{
				const std::lock_guard lock(mutex);
				end = std::chrono::steady_clock::now();
				complete = true;
			}
			completed.notify_all();
		}
	};

} // namespace detail

copy_handle::copy_handle(std::shared_ptr<detail::copy_state> state) : state(std::move(state)) {}

bool copy_handle::is_complete() const { return state->complete; }

void copy_handle::wait() const {
	std::unique_lock lock(state->mutex);
	state->completed.wait(lock, [this] { return state->complete.load(); });
}

std::optional<std::string> copy_handle::error() const {
	if(!is_complete()) { return std::nullopt; }
	const std::lock_guard lock(state->mutex);
	return state->error;
}

std::optional<std::chrono::nanoseconds> copy_handle::execution_time() const {
	if(!is_complete()) { return std::nullopt; }
	return std::chrono::duration_cast<std::chrono::nanoseconds>(state->end - state->start);
}

copy_handle execute_copy(executor& exec, const parallel_copy_set& set) {
	COPYLIB_ENSURE(is_valid(set), "Invalid copy set: {}", set);
	if(std::getenv("COPYLIB_TRACE") && std::atoi(std::getenv("COPYLIB_TRACE"))) { utils::err_print("copylib trace: copy of {} plans\n", set.size()); }
	// every plan has to fit into one worker's slice of staging; checked here, so that a plan too large throws from this call
	for(const auto& plan : set) {
		detail::staging_fulfiller fits(exec, 0);
		for(auto spec : plan) {
			fits.fulfill(spec);
		}
	}
	auto state = std::make_shared<detail::copy_state>();

	const int64_t total_plans = set.size();
	state->unfinished_plans = total_plans;
	if(total_plans == 0) {
		state->finish_call();
		return copy_handle(state);
	}

	// one contiguous part of the plans per queue index, each run by one worker; single-copy plans alternate devices
	const int64_t parts_count = exec.get_queues_per_device();
	int64_t first_plan = 0;
	const char* order = std::getenv("COPYLIB_PLAN_ORDER");
	const bool dynamic = order && !std::strcmp(order, "dynamic");
	const auto next_plan = std::make_shared<std::atomic<int64_t>>(0);
	for(int64_t part = 0; part < parts_count; part++) {
		const int64_t plans_in_part = total_plans / parts_count + (part < total_plans % parts_count ? 1 : 0);
		if(plans_in_part == 0) { continue; }
		// the worker owns its plans and shares the state, since the call returns before the plans have run
		std::vector<copy_plan> plans;
		for(int64_t i = 0; i < (dynamic ? total_plans : plans_in_part); i++) {
			plans.push_back(set[dynamic ? i : order && !std::strcmp(order, "interleaved") ? part + i * parts_count : first_plan + i]);
		}
		first_plan += plans_in_part;
		exec.pool.detach_task([&exec, state, part, plans = std::move(plans), dynamic, next_plan] {
			// staging goes into this worker's own slice, from its start for every plan: the worker runs one plan at a time and
			// waits for it to finish, and no other worker uses the slice, so calls in flight together never share staging
			const auto slice = static_cast<int64_t>(BS::this_thread::get_index().value());
			int64_t plan_idx = 0;
			for(int64_t i = dynamic ? (*next_plan)++ : 0; i < static_cast<int64_t>(plans.size()); i = dynamic ? (*next_plan)++ : i + 1) {
				const auto& plan = plans[i];
				const bool use_alternate_device = plan.size() == 1 && plan_idx % 2 == 1;
				try {
					detail::staging_fulfiller fulfiller(exec, slice);
					execute_plan_impl(exec, plan, fulfiller, part, use_alternate_device);
				} catch(const std::exception& e) { //
					state->fail(e.what());
				} catch(...) { state->fail("unknown exception"); }
				plan_idx++;
				state->finish_plan();
			}
		});
	}
	return copy_handle(state);
}

} // namespace copylib
