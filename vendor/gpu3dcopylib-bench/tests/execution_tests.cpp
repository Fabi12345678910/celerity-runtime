#include "backend_test_utils.hpp"

#include <copylib/backend.hpp>
#include <copylib/core.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// Layers 10 and 11: several copies in flight, and the semantics of copy_handle.
//
// Everything here uses real copies; there is deliberately no hook into the workers. So a copy that is not complete yet
// cannot be asserted, since SimSYCL may already have finished it, and ordering between the steps of a plan is left to
// the layer 9 bytes on an asynchronous SYCL implementation. See docs/testing.md.

using namespace copylib;
using namespace copylib_testing;
using Catch::Matchers::ContainsSubstring;
namespace ref = copylib_testing::reference_box;

namespace {

const copy_strategy direct_strategy = strategy_from_fields(copy_type::direct, copy_properties::none, d2d_implementation::direct, 0);
const copy_strategy chunked_strategy = strategy_from_fields(copy_type::direct, copy_properties::use_kernel, d2d_implementation::direct, 64);
const copy_strategy staged_strategy = strategy_from_fields(copy_type::staged, copy_properties::none, d2d_implementation::direct, 0);
const copy_strategy staged_chunked_strategy = strategy_from_fields(copy_type::staged, copy_properties::use_kernel, d2d_implementation::direct, 64);
const copy_strategy host_staged_chunked_strategy = strategy_from_fields(copy_type::staged, copy_properties::none, d2d_implementation::host_staging_at_both, 64);

copy_handle launch(executor& exec, const prepared_copy& copy, const copy_strategy& strategy) {
	return execute_copy(exec, manifest_strategy(copy.spec(), strategy, basic_staging_provider{}));
}

// several MiB in one contiguous run, so that the copy is likely, though not certain, to still run right after the call
data_layout large_run() { return normal_form::one_run(0, 0, int64_t{8} << 20); }

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// Layer 10: calls in flight

const std::vector<std::pair<location, location>> in_flight_ends = {
    {on_device(device_id::d0), on_device(device_id::d1)},
    {on_device(device_id::d1), on_device(device_id::d0)},
    {on_device(device_id::d0), on_device(device_id::d0)},
    {pinned_host, on_device(device_id::d1)},
    {on_device(device_id::d0), pageable_host},
    {pageable_host, pageable_host},
};

// launches every end pair `rounds` times with alternating strategies before waiting for any of them
void check_calls_in_flight(executor& exec, const copy_strategy& even, const copy_strategy& odd, int rounds) {
	std::vector<std::unique_ptr<prepared_copy>> copies;
	std::vector<copy_handle> handles;
	std::vector<std::string> names;
	for(int round = 0; round < rounds; ++round) {
		for(size_t i = 0; i < in_flight_ends.size(); ++i) {
			const auto& [from, to] = in_flight_ends[i];
			copies.push_back(std::make_unique<prepared_copy>(exec, from, ref::fields(), to, shapes::six_rows_of_48(0)));
			handles.push_back(launch(exec, *copies.back(), i % 2 == 0 ? even : odd));
			names.push_back(describe(from) + " -> " + describe(to));
		}
	}

	copy_report report;
	for(size_t i = 0; i < copies.size(); ++i) {
		handles[i].wait();
		report.add(copies[i]->verify(handles[i].error()), names[i]);
	}
	report.check();
}

TEST_CASE("several calls in flight each produce correct bytes", "[execution][concurrency]") {
	auto exec = make_executor();
	check_calls_in_flight(exec, direct_strategy, chunked_strategy, 1);
}

TEST_CASE("staged calls in flight at the same time never share staging memory", "[execution][concurrency][staging]") {
	// every worker stages in its own slice, fresh for each plan; overlapping calls would otherwise overwrite each other's
	// staged chunks
	auto exec = make_executor();
	check_calls_in_flight(exec, staged_chunked_strategy, host_staged_chunked_strategy, 4);
}

TEST_CASE("a chunked copy may stage more than the buffer size in total", "[execution][staging]") {
	// 288 bytes in chunks of 64 stage 5 x 128 bytes on each device, more than the 256 bytes there are; one chunk fits a slice
	executor exec(256, test_devices(2), 1);
	REQUIRE(exec.get_staging_slice_size() == 256);
	const prepared_copy copy(exec, on_device(device_id::d0), ref::fields(), on_device(device_id::d1), shapes::six_rows_of_48(0));

	const auto outcome = run_copy(exec, copy, staged_chunked_strategy);
	INFO(outcome.describe());
	CHECK(outcome.correct());
}

TEST_CASE("a plan staging more than a worker's slice is rejected by the call", "[execution][staging][error]") {
	// two workers share 256 bytes, so each slice holds 128 bytes; an unchunked staged copy of 288 bytes cannot fit
	executor exec(256, test_devices(2), 2);
	REQUIRE(exec.get_staging_slice_size() == 128);
	const prepared_copy copy(exec, on_device(device_id::d0), ref::fields(), on_device(device_id::d1), shapes::six_rows_of_48(0));

	CHECK_THROWS_AS(launch(exec, copy, staged_strategy), copylib::error);
	CHECK_NOTHROW(launch(exec, copy, staged_chunked_strategy).wait());
}

// ---------------------------------------------------------------------------------------------------------------------
// Layer 11: the handle

TEST_CASE("once wait returns, the handle reports a completed copy", "[execution][handle]") {
	auto exec = make_executor();
	const prepared_copy copy(exec, on_device(device_id::d0), ref::fields(), on_device(device_id::d1), ref::fields());

	const auto handle = launch(exec, copy, staged_strategy);
	handle.wait();

	CHECK(handle.is_complete());
	CHECK(handle.execution_time().has_value());
	CHECK_FALSE(handle.error().has_value());
	const auto outcome = copy.verify(handle.error());
	INFO(outcome.describe());
	CHECK(outcome.correct());
}

TEST_CASE("is_complete never goes back from true to false", "[execution][handle]") {
	auto exec = make_executor();
	const prepared_copy copy(exec, on_device(device_id::d0), large_run(), on_device(device_id::d1), large_run());

	const auto handle = launch(exec, copy, direct_strategy);
	bool seen_complete = false;
	bool went_back = false;
	for(int i = 0; i < 10000; ++i) {
		const bool complete = handle.is_complete();
		went_back = went_back || (seen_complete && !complete);
		seen_complete = seen_complete || complete;
	}
	handle.wait();
	for(int i = 0; i < 1000; ++i) {
		went_back = went_back || !handle.is_complete();
	}

	CHECK_FALSE(went_back);
	CHECK(handle.is_complete());
	CHECK(copy.verify(handle.error()).correct());
}

TEST_CASE("is_complete returns immediately while a copy runs", "[execution][handle]") {
	// a generous bound: is_complete is a single atomic load, so anything close to this means it waited on the copy
	constexpr auto bound = std::chrono::milliseconds(50);
	auto exec = make_executor();
	const prepared_copy copy(exec, on_device(device_id::d0), large_run(), on_device(device_id::d1), large_run());

	const auto handle = launch(exec, copy, direct_strategy);
	auto slowest = std::chrono::steady_clock::duration::zero();
	for(int i = 0; i < 100; ++i) {
		const auto before = std::chrono::steady_clock::now();
		static_cast<void>(handle.is_complete());
		slowest = std::max(slowest, std::chrono::steady_clock::now() - before);
	}
	handle.wait();

	CHECK(slowest < bound);
	CHECK(handle.is_complete());
	CHECK(copy.verify(handle.error()).correct());
}

TEST_CASE("copies of a handle share one state", "[execution][handle]") {
	auto exec = make_executor();
	const prepared_copy copy(exec, pinned_host, ref::fields(), on_device(device_id::d0), ref::fields());

	const auto original = launch(exec, copy, direct_strategy);
	const auto duplicate = original; // NOLINT(performance-unnecessary-copy-initialization): the copy is the point
	original.wait();

	CHECK(duplicate.is_complete());
	CHECK(duplicate.execution_time() == original.execution_time());
	CHECK(duplicate.execution_time().has_value());
}

TEST_CASE("handles can be waited on in any order", "[execution][handle]") {
	auto exec = make_executor();
	std::vector<std::unique_ptr<prepared_copy>> copies;
	std::vector<copy_handle> handles;
	for(int i = 0; i < 3; ++i) {
		copies.push_back(std::make_unique<prepared_copy>(exec, on_device(device_id::d0), ref::fields(), on_device(device_id::d1), ref::fields()));
		handles.push_back(launch(exec, *copies.back(), chunked_strategy));
	}

	copy_report report;
	for(size_t i = copies.size(); i-- > 0;) {
		handles[i].wait();
		CHECK(handles[i].is_complete());
		report.add(copies[i]->verify(handles[i].error()), "copy " + std::to_string(i));
	}
	report.check();
}

TEST_CASE("dropping the handle and destroying the executor still completes the copy", "[execution][handle]") {
	// pageable host memory on both ends, the only memory that stays usable after the executor and its queues are gone
	configure_test_system();
	auto exec = std::make_unique<executor>(test_staging_bytes, test_devices(2), 2);
	const prepared_copy copy(*exec, pageable_host, ref::fields(), pageable_host, shapes::six_rows_of_48(0));

	static_cast<void>(launch(*exec, copy, chunked_strategy));
	exec.reset(); // waits for every copy still in flight

	const auto outcome = copy.verify(std::nullopt);
	INFO(outcome.describe());
	CHECK(outcome.correct());
}

TEST_CASE("a plan failing in its worker is reported through the handle while the others complete", "[execution][handle][error]") {
	// a natural failure inside a worker: the plan names a device the executor does not have, which no check on the
	// caller's thread looks at; its memory is pageable, so that nothing touches device memory should it run anyway
	auto exec = make_executor();
	const prepared_copy unknown(exec, pageable_host, ref::fields(), pageable_host, ref::fields());
	auto failing = unknown.spec();
	failing.source_device = failing.target_device = device_id::d5;
	const prepared_copy copy(exec, on_device(device_id::d0), ref::fields(), on_device(device_id::d1), ref::fields());

	const auto handle = execute_copy(exec, parallel_copy_set{copy_plan{failing}, copy_plan{copy.spec()}});
	handle.wait();

	CHECK(handle.is_complete());
	REQUIRE(handle.error().has_value());
	CHECK_THAT(*handle.error(), ContainsSubstring("Invalid device id"));
	const auto outcome = copy.verify(std::nullopt);
	INFO(outcome.describe());
	CHECK(outcome.correct());
}

TEST_CASE("execute_copy rejects an invalid set", "[execution][error]") {
	// window lengths differ, but every window stays inside its allocation, so a missing check cannot corrupt memory
	auto exec = make_executor();
	const prepared_copy copy(exec, on_device(device_id::d0), ref::fields(), on_device(device_id::d1), ref::fields());
	auto invalid = copy.spec();
	invalid.target_layout.end -= 8;
	REQUIRE_FALSE(is_valid(invalid));

	CHECK_THROWS_AS(static_cast<void>(execute_copy(exec, parallel_copy_set{copy_plan{invalid}})), copylib::error);
	CHECK_NOTHROW(execute_copy(exec, parallel_copy_set{copy_plan{copy.spec()}}).wait());
}
