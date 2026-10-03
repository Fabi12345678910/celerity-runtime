#pragma once

#include "test_utils.hpp"

#include <copylib/backend.hpp>
#include <copylib/support.hpp>

#ifdef SIMSYCL_VERSION
#include <simsycl/system.hh>
#endif

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// Harness for the backend layers 9-12. The tests own every byte they copy: the library only copies memory it is given,
// so allocation, filling and checking happen here.

namespace copylib_testing {

// ---------------------------------------------------------------------------------------------------------------------
// Test system and executor

// The harness, not the library, decides which devices exist: under SimSYCL four identical GPUs, as the 2D executor
// configured for itself. This has to happen before the first device query, so every executor is built after calling it.
inline void configure_test_system() {
#ifdef SIMSYCL_VERSION
	[[maybe_unused]] static const bool configured = [] {
		auto config = simsycl::get_default_system_config();
		config.devices.emplace("gpu2", config.devices.cbegin()->second);
		config.devices.emplace("gpu3", config.devices.cbegin()->second);
		config.devices.emplace("gpu4", config.devices.cbegin()->second);
		simsycl::configure_system(config);
		return true;
	}();
#endif
}

// staging memory per device, generous for the small copies of these tests
inline constexpr int64_t test_staging_bytes = int64_t{1} << 20;

// The devices the tests copy between: the first `count` GPUs, as SimSYCL provides in CI. Where there are fewer, the first
// device is repeated, so that a machine with a single device, such as AdaptiveCpp's CPU device, runs every test as well.
[[nodiscard]] inline std::vector<sycl::device> test_devices(int64_t count) {
	configure_test_system();
	auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
	if(devices.empty()) { devices = sycl::device::get_devices(); }
	devices.resize(std::min(devices.size(), static_cast<size_t>(count)));
	while(devices.size() < static_cast<size_t>(count)) {
		devices.push_back(devices.front());
	}
	return devices;
}

[[nodiscard]] inline copylib::executor make_executor(int64_t devices = 2, int64_t queues_per_device = 2, int64_t staging_bytes = test_staging_bytes) {
	return copylib::executor(staging_bytes, test_devices(devices), queues_per_device);
}

// ---------------------------------------------------------------------------------------------------------------------
// Memory owned by the tests

enum class memory_kind { device, pinned_host, pageable_host };

// where one end of a copy lives; host memory has device_id::host
struct location {
	copylib::device_id did;
	memory_kind kind;
	int64_t misalignment = 0; // bytes past a 64-byte boundary at which the memory starts
};

[[nodiscard]] inline constexpr location on_device(copylib::device_id did) { return {did, memory_kind::device}; }
inline constexpr location pinned_host{copylib::device_id::host, memory_kind::pinned_host};
inline constexpr location pageable_host{copylib::device_id::host, memory_kind::pageable_host};

[[nodiscard]] inline std::string describe(const location& where) {
	switch(where.kind) {
	case memory_kind::device: return copylib::utils::format("{} memory", where.did);
	case memory_kind::pinned_host: return "pinned host memory";
	case memory_kind::pageable_host: return "pageable host memory";
	}
	return {};
}

// One allocation, starting `where.misalignment` bytes past a 64-byte boundary. Pinned host memory is allocated through
// the queue of `context_device`, the device at the other end of the copy, as the 2D executor kept a host buffer per device.
class test_allocation {
  public:
	test_allocation(copylib::executor& exec, location where, int64_t size, copylib::device_id context_device) : where(where), bytes(size) {
		// aligned allocations must be a whole number of alignments; SimSYCL aborts otherwise
		const auto allocated = static_cast<size_t>((where.misalignment + size + alignment - 1) / alignment * alignment);
		switch(where.kind) {
		case memory_kind::device:
			queue = &exec.get_queue(where.did);
			raw = sycl::aligned_alloc_device<std::byte>(alignment, allocated, *queue);
			break;
		case memory_kind::pinned_host:
			queue = &exec.get_queue(context_device);
			raw = sycl::aligned_alloc_host<std::byte>(alignment, allocated, *queue);
			break;
		case memory_kind::pageable_host: raw = static_cast<std::byte*>(std::aligned_alloc(alignment, allocated)); break;
		}
		if(raw == nullptr) { throw std::bad_alloc(); }
		ptr = raw + where.misalignment;
	}

	test_allocation(const test_allocation&) = delete;
	test_allocation& operator=(const test_allocation&) = delete;

	~test_allocation() {
		if(where.kind == memory_kind::pageable_host) {
			std::free(raw);
		} else {
			sycl::free(raw, *queue);
		}
	}

	[[nodiscard]] intptr_t base() const { return reinterpret_cast<intptr_t>(ptr); }
	[[nodiscard]] int64_t size() const { return bytes; }

	void write(const std::vector<std::byte>& data) {
		if(where.kind == memory_kind::device) {
			queue->memcpy(ptr, data.data(), data.size()).wait();
		} else {
			std::memcpy(ptr, data.data(), data.size());
		}
	}

	[[nodiscard]] std::vector<std::byte> read() const {
		std::vector<std::byte> data(static_cast<size_t>(bytes));
		if(where.kind == memory_kind::device) {
			queue->memcpy(data.data(), ptr, data.size()).wait();
		} else {
			std::memcpy(data.data(), ptr, data.size());
		}
		return data;
	}

  private:
	static constexpr size_t alignment = 64;

	location where;
	int64_t bytes;
	sycl::queue* queue = nullptr;
	std::byte* raw = nullptr; // as allocated, `where.misalignment` bytes before ptr
	std::byte* ptr = nullptr;
};

// ---------------------------------------------------------------------------------------------------------------------
// Reference copy

// what the target holds wherever the copy must not write
inline constexpr std::byte sentinel{0xcd};

// a source pattern with a period that is not a power of two, so that shifted or misaligned copies do not match by chance
[[nodiscard]] inline std::vector<std::byte> source_pattern(int64_t size) {
	std::vector<std::byte> data(static_cast<size_t>(size));
	for(int64_t i = 0; i < size; ++i) {
		data[static_cast<size_t>(i)] = static_cast<std::byte>((i * 131 + 17) % 251);
	}
	return data;
}

// the allocation a layout needs: up to the end of its box, plus slack behind it to catch writes past the box
[[nodiscard]] inline int64_t allocation_size(const copylib::data_layout& layout) {
	constexpr int64_t slack = 64;
	const int64_t box_bytes =
	    (layout.d0_end_offset - layout.d0_start_offset) * (layout.d1_end_offset - layout.d1_start_offset) * (layout.d2_end_offset - layout.d2_start_offset);
	const auto offsets = reference_offsets(with_window_fields(layout, 0, box_bytes));
	return (offsets.empty() ? 0 : offsets.back() + 1) + slack;
}

// The whole target allocation as the oracle expects it after the copy: the sentinel everywhere, except the target
// window, which holds the source bytes the windows map there.
[[nodiscard]] inline std::vector<std::byte> expected_target(
    const std::vector<std::byte>& source_bytes, const copylib::data_layout& source, int64_t target_size, const copylib::data_layout& target) {
	std::vector<std::byte> expected(static_cast<size_t>(target_size), sentinel);
	const auto from = reference_offsets(source);
	const auto to = reference_offsets(target);
	for(size_t i = 0; i < from.size() && i < to.size(); ++i) {
		expected[static_cast<size_t>(to[i])] = source_bytes[static_cast<size_t>(from[i])];
	}
	return expected;
}

// index of the first differing byte, or -1; comparing the vectors directly would print every byte on failure
[[nodiscard]] inline int64_t first_mismatch(const std::vector<std::byte>& actual, const std::vector<std::byte>& expected) {
	if(actual.size() != expected.size()) return static_cast<int64_t>(std::min(actual.size(), expected.size()));
	for(size_t i = 0; i < actual.size(); ++i) {
		if(actual[i] != expected[i]) return static_cast<int64_t>(i);
	}
	return -1;
}

struct copy_outcome {
	std::optional<std::string> error; // reported by the handle, or thrown
	int64_t wrong_target_byte = -1;   // first target byte that differs from the reference, -1 if none
	int64_t changed_source_byte = -1; // first source byte the copy changed, -1 if none

	[[nodiscard]] bool correct() const { return !error.has_value() && wrong_target_byte == -1 && changed_source_byte == -1; }

	[[nodiscard]] std::string describe() const {
		if(error.has_value()) return "error: " + *error;
		if(wrong_target_byte != -1) return copylib::utils::format("target differs from the reference at byte {}", wrong_target_byte);
		if(changed_source_byte != -1) return copylib::utils::format("source changed at byte {}", changed_source_byte);
		return "correct";
	}
};

// One copy between fresh allocations. The layouts' own bases are replaced by the allocations', and the source is filled
// with a pattern and the target with the sentinel.
class prepared_copy {
  public:
	prepared_copy(copylib::executor& exec, location from, const copylib::data_layout& source, location to, const copylib::data_layout& target)
	    : source_memory(exec, from, allocation_size(source), context_for(from, to)), target_memory(exec, to, allocation_size(target), context_for(to, from)),
	      source_bytes(source_pattern(source_memory.size())),
	      copy(spec_from_fields(from.did, placed(source, source_memory), to.did, placed(target, target_memory))) {
		source_memory.write(source_bytes);
		target_memory.write(std::vector<std::byte>(static_cast<size_t>(target_memory.size()), sentinel));
	}

	[[nodiscard]] const copylib::copy_spec& spec() const { return copy; }

	[[nodiscard]] copy_outcome verify(std::optional<std::string> error) const {
		copy_outcome outcome;
		outcome.error = std::move(error);
		outcome.wrong_target_byte =
		    first_mismatch(target_memory.read(), expected_target(source_bytes, copy.source_layout, target_memory.size(), copy.target_layout));
		outcome.changed_source_byte = first_mismatch(source_memory.read(), source_bytes);
		return outcome;
	}

  private:
	static copylib::device_id context_for(location self, location other) {
		if(self.kind != memory_kind::pinned_host) return self.did;
		return other.kind == memory_kind::device ? other.did : copylib::device_id::d0;
	}

	static copylib::data_layout placed(copylib::data_layout layout, const test_allocation& memory) {
		layout.base = memory.base();
		return layout;
	}

	test_allocation source_memory;
	test_allocation target_memory;
	std::vector<std::byte> source_bytes;
	copylib::copy_spec copy;
};

// Manifests the strategy, runs the set and waits. Exceptions count as reported errors, so that a failing copy is
// recorded rather than ending the test case.
[[nodiscard]] inline copy_outcome run_copy(copylib::executor& exec, const prepared_copy& copy, const copylib::copy_strategy& strategy) {
	std::optional<std::string> error;
	try {
		const auto set = copylib::manifest_strategy(copy.spec(), strategy, copylib::basic_staging_provider{});
		const auto handle = copylib::execute_copy(exec, set);
		handle.wait();
		error = handle.error();
	} catch(const std::exception& e) { error = e.what(); }
	return copy.verify(std::move(error));
}

// Counts wrong copies and reports only the first, so that a broken backend cannot flood the CI log.
struct copy_report {
	int copies = 0;
	int failures = 0;
	std::string first_failure;

	void add(const copy_outcome& outcome, const std::string& where) {
		++copies;
		if(!outcome.correct() && failures++ == 0) first_failure = where + ": " + outcome.describe();
	}

	void check() const {
		INFO(failures << " of " << copies << " copies are wrong, the first being " << first_failure);
		CHECK(copies > 0);
		CHECK(failures == 0);
	}
};

} // namespace copylib_testing
