#pragma once

#include "core.hpp"

#include <sycl/sycl.hpp>

#include <bs_thread_pool/bs_thread_pool.hpp>

#include <chrono>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace copylib {

namespace detail {

	// what the executor keeps per device: its queues and its staging memory
	struct device {
		sycl::device dev;
		std::vector<sycl::queue> queues;
		std::byte* staging_buffer = nullptr;
		std::byte* host_staging_buffer = nullptr;
		std::optional<int> host_staging_cpu; // the CPU the host staging was allocated from, if pinned

		device(sycl::device dev, const std::vector<sycl::queue>& queues);
		device(const device&) = delete;
		device& operator=(const device&) = delete;

		~device();
	};

	// a deque never relocates its elements, so a device and the buffers it frees exist exactly once
	using device_list = std::deque<device>;

} // namespace detail

class copy_handle;

class executor {
  public:
	struct target {
		device_id did;
		int64_t queue_idx;

		bool operator==(const target& other) const = default;
		bool operator!=(const target& other) const = default;
	};
	static constexpr target null_target = target{device_id::count, 0};

	// device_contexts[i] becomes d_i. Its queues and staging memory are created in the given context, which has to contain
	// the device and be the one the caller's memory on that device belongs to.
	executor(int64_t buffer_size, const std::vector<std::pair<sycl::device, sycl::context>>& device_contexts, int64_t queues_per_device = 1);
	// devices[i] becomes d_i, in the default context that a queue created from the device alone gets
	executor(int64_t buffer_size, const std::vector<sycl::device>& devices, int64_t queues_per_device = 1);
	// the first devices_needed GPUs, in their default contexts
	executor(int64_t buffer_size, int64_t devices_needed, int64_t queues_per_device = 1);
	// every GPU, with one queue each
	explicit executor(int64_t buffer_size);

	sycl::queue& get_queue(device_id id, int64_t queue_idx = 0);
	sycl::queue& get_queue(const target& tgt);

	std::byte* get_staging_buffer(device_id id);
	std::byte* get_host_staging_buffer(device_id id);

	[[nodiscard]] int64_t get_buffer_size() const;
	[[nodiscard]] int64_t get_queues_per_device() const;
	// the staging memory of each worker, per device and kind: the buffer size divided among the queues_per_device workers,
	// rounded down to the staging alignment; the staging of every plan has to fit into it
	[[nodiscard]] int64_t get_staging_slice_size() const;

	[[nodiscard]] std::string get_sycl_impl_name() const;
	[[nodiscard]] bool is_device_to_device_copy_available() const;
	[[nodiscard]] bool is_peer_memory_access_available() const;
	[[nodiscard]] int32_t get_preferred_wg_size() const;
	[[nodiscard]] std::string get_info() const;

	enum class possibility {
		possible,
		needs_d2d_copy,
	};

	[[nodiscard]] possibility can_copy(const copy_spec& spec) const;
	[[nodiscard]] possibility can_copy(const parallel_copy_set& set) const;

	void barrier();

  private:
	friend copy_handle execute_copy(executor& exec, const parallel_copy_set& set);

	detail::device_list devices;
	int64_t buffer_size;
	bool peer_access_available = false; // enabled and checked once by the constructor
	int64_t staging_slice_size = 0;
	int32_t preferred_wg_size = 0;

	// declared last, so it is destroyed first: waits for the copies in flight while queues and staging memory still exist
	BS::light_thread_pool pool;
};

// The CPU to allocate the host staging of device gpu_idx from, taken from COPYLIB_ALLOC_CPU_IDS (one CPU ID per device,
// comma-separated). Empty when the variable is not set: the right CPUs depend on the machine and on the CPUs a job may
// use, so host staging is pinned only on request.
std::optional<int> get_cpu_for_gpu_alloc(int gpu_idx, size_t total_gpu_count);

namespace detail {

	// alignment of the staging buffers and of every offset handed out in them
	inline constexpr int64_t staging_alignment = 128;

	sycl::event copy_with_kernel(sycl::queue& q, const copy_spec& spec, int32_t preferred_wg_size);

	class staging_fulfiller {
	  public:
		// places staging in the slice of worker `slice`, from its start; one fulfiller serves one plan
		staging_fulfiller(executor& exec, int64_t slice);

		void fulfill(data_layout& layout);
		void fulfill(copy_spec& spec);

	  private:
		struct staging_info {
			int64_t size = 0;
			device_id device = device_id::d0;
			bool on_host = false;
			std::byte* buffer = nullptr;
		};

		executor& exec;
		int64_t slice_offset;
		std::vector<int64_t> current_staging_offsets = std::vector<int64_t>(static_cast<int>(device_id::count), 0);
		std::vector<int64_t> current_host_staging_offsets = std::vector<int64_t>(static_cast<int>(device_id::count), 0);

		std::unordered_map<decltype(staging_id::index), staging_info> staging_buffers;
	};

	struct copy_state; // shared by a handle and the workers running its plans

	// where a step of a plan ran and the last command it submitted there, which the next step waits on when it runs
	// elsewhere; the event of a host step is default-constructed and therefore complete
	struct step_result {
		executor::target target = executor::null_target;
		sycl::event event;
	};

	step_result execute_copy(executor& exec, const copy_spec& spec, int64_t queue_idx = 0, bool alternate_device = false, step_result last = {});

} // namespace detail

// Completion state of one execute_copy call. Copies of a handle share the same state.
class copy_handle {
  public:
	// true once every plan has finished, successfully or not; monotonic, never blocks, never throws
	[[nodiscard]] bool is_complete() const;
	// blocks until is_complete() is true; never throws
	void wait() const;

	// the first failure of any plan once complete, empty on success
	[[nodiscard]] std::optional<std::string> error() const;
	// time from the call until the last plan finished, once complete
	[[nodiscard]] std::optional<std::chrono::nanoseconds> execution_time() const;

  private:
	explicit copy_handle(std::shared_ptr<detail::copy_state> state);
	friend copy_handle execute_copy(executor& exec, const parallel_copy_set& set);

	std::shared_ptr<detail::copy_state> state;
};

// hands the plans of the set to the executor's thread pool and returns immediately
[[nodiscard]] copy_handle execute_copy(executor& exec, const parallel_copy_set& set);

} // namespace copylib
