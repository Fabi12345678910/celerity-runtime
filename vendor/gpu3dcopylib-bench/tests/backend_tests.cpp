#include "backend_test_utils.hpp"

#include <copylib/backend.hpp>
#include <copylib/core.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstdlib>
#include <random>
#include <string>
#include <utility>
#include <vector>

// Layer 9: execute_copy produces exactly the bytes of a host-side reference copy.
//
// Each case copies between fresh allocations: the source holds a pattern, the target a sentinel, and afterwards the whole
// target allocation must equal the reference, which also catches writes outside the window, and the source must be
// unchanged. Every case runs over all pairs of ends in the endpoint list, host ends in pinned and pageable memory, and
// over a set of strategies covering each copy type, kernels on and off, chunking and every d2d implementation. A copy
// with a pageable host end must not take the kernel path even when the strategy asks for kernels.
//
// On SimSYCL this checks the bytes only. Run on an asynchronous SYCL implementation, correct bytes also stand in for
// correct ordering between the steps of a plan (see docs/testing.md, layer 10).

using namespace copylib;
using namespace copylib_testing;
namespace ref = copylib_testing::reference_box;

namespace {

std::vector<copy_strategy> strategies() {
	return {
	    strategy_from_fields(copy_type::direct, copy_properties::none, d2d_implementation::direct, 0),
	    strategy_from_fields(copy_type::direct, copy_properties::use_kernel, d2d_implementation::direct, 0),
	    strategy_from_fields(copy_type::direct, copy_properties::none, d2d_implementation::direct, 64),
	    strategy_from_fields(copy_type::staged, copy_properties::none, d2d_implementation::direct, 0),
	    strategy_from_fields(copy_type::staged, copy_properties::use_kernel, d2d_implementation::direct, 96),
	    strategy_from_fields(copy_type::staged, copy_properties::use_kernel, d2d_implementation::host_staging_at_source, 0),
	    strategy_from_fields(copy_type::direct, copy_properties::none, d2d_implementation::host_staging_at_target, 40),
	    strategy_from_fields(copy_type::staged, copy_properties::none, d2d_implementation::host_staging_at_both, 32),
	};
}

// every pair of ends the executor has to handle, host ends in both kinds of host memory
std::vector<std::pair<location, location>> endpoint_pairs() {
	return {
	    {on_device(device_id::d0), on_device(device_id::d0)},
	    {on_device(device_id::d0), on_device(device_id::d1)},
	    {pinned_host, on_device(device_id::d1)},
	    {on_device(device_id::d0), pinned_host},
	    {pageable_host, on_device(device_id::d0)},
	    {on_device(device_id::d1), pageable_host},
	    {pinned_host, pageable_host},
	    {pageable_host, pageable_host},
	};
}

std::string where(const location& from, const location& to, const copy_strategy& strategy) {
	return describe(from) + " -> " + describe(to) + " with " + utils::format("{}", strategy);
}

// copies source to target for every pair of ends and every strategy, in memory starting the given number of bytes past a
// 64-byte boundary
void check_all(const data_layout& source, const data_layout& target, int64_t source_misalignment = 0, int64_t target_misalignment = 0) {
	auto exec = make_executor();
	copy_report report;
	for(auto [from, to] : endpoint_pairs()) {
		from.misalignment = source_misalignment;
		to.misalignment = target_misalignment;
		for(const auto& strategy : strategies()) {
			const prepared_copy copy(exec, from, source, to, target);
			report.add(run_copy(exec, copy, strategy), where(from, to, strategy));
		}
	}
	report.check();
}

data_layout reference_window(int64_t start, int64_t end) { return with_window_fields(ref::fields(), start, end); }

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// Forced cases

TEST_CASE("the reference box is copied into the same box of another allocation", "[backend]") { check_all(ref::fields(), ref::fields()); }

TEST_CASE("a whole allocation is copied as one run", "[backend]") { check_all(shapes::whole_allocation(), shapes::whole_allocation()); }

TEST_CASE("a single partial row is copied", "[backend]") { check_all(shapes::single_row(), shapes::single_row()); }

TEST_CASE("a window inside one row is copied", "[backend][window]") { check_all(reference_window(2, 10), reference_window(2, 10)); }

TEST_CASE("a window crossing row and plane boundaries is copied", "[backend][window]") {
	// a plane of the box holds 96 bytes, so [12, 200) starts mid-row and ends in the third plane
	check_all(reference_window(12, 200), reference_window(12, 200));
}

TEST_CASE("a single byte is copied", "[backend][window]") { check_all(reference_window(5, 6), reference_window(5, 6)); }

TEST_CASE("windows at different offsets of either side are copied", "[backend][window]") {
	// shifted by 8 bytes, which limits the alignment to 8
	check_all(reference_window(0, 280), reference_window(8, 288));
}

TEST_CASE("a reshaping copy is copied", "[backend]") {
	// rows of 24 bytes into rows of 48
	check_all(ref::fields(), shapes::six_rows_of_48(0));
}

TEST_CASE("a strided box is copied from and into a contiguous run", "[backend]") {
	check_all(ref::fields(), normal_form::one_run(0, 0, ref::total_bytes));
	check_all(normal_form::one_run(0, 0, ref::total_bytes), ref::fields());
}

TEST_CASE("odd row extents are copied byte by byte", "[backend][alignment]") {
	// 9-byte rows 13 bytes apart: alignment 1
	const auto odd = layout_from_fields(0, 13, 6, 0, 0, 0, 9, 5, 2, 0, 9 * 5 * 2);
	REQUIRE(copy_alignment(spec_from_fields(device_id::d0, odd, device_id::d1, odd)) == 1);
	check_all(odd, odd);
}

TEST_CASE("rows aligned to 64 bytes are copied", "[backend][alignment]") {
	// 128-byte rows starting at byte 64 of 256-byte rows: alignment 64, the widest element the kernels use
	const auto wide = layout_from_fields(0, 256, 4, 64, 0, 0, 192, 4, 2, 0, 128 * 4 * 2);
	REQUIRE(copy_alignment(spec_from_fields(device_id::d0, wide, device_id::d1, wide)) == 64);
	check_all(wide, wide);
}

TEST_CASE("rows aligned to more than their memory are copied", "[backend][alignment]") {
	// the rows above, but the source starts 2 bytes past a 64-byte boundary, the least the library allows, and the target
	// 8 bytes, so the kernels have to narrow their element to the bases
	const auto wide = layout_from_fields(0, 256, 4, 64, 0, 0, 192, 4, 2, 0, 128 * 4 * 2);
	check_all(wide, wide, 2, 8);
}

// ---------------------------------------------------------------------------------------------------------------------
// Randomized

namespace {

// Uniform in [lo, hi], by plain modulo so that a seed reproduces the same cases with every standard library.
int64_t pick(std::mt19937_64& rng, int64_t lo, int64_t hi) { return lo + static_cast<int64_t>(rng() % static_cast<uint64_t>(hi - lo + 1)); }

// a random box in an allocation of up to 64-byte rows, 5 rows per plane and 6 planes, with its window over the whole box
data_layout random_box(std::mt19937_64& rng) {
	const int64_t elem = int64_t{1} << pick(rng, 0, 3);
	const int64_t row_elems = pick(rng, 1, 8);
	const int64_t d0_start = pick(rng, 0, row_elems - 1);
	const int64_t d0_end = pick(rng, d0_start + 1, row_elems);
	const int64_t d1_stride = pick(rng, 1, 5);
	const int64_t d1_start = pick(rng, 0, d1_stride - 1);
	const int64_t d1_end = pick(rng, d1_start + 1, d1_stride);
	const int64_t d2_start = pick(rng, 0, 3);
	const int64_t d2_end = d2_start + pick(rng, 1, 3);
	const int64_t total = (d0_end - d0_start) * elem * (d1_end - d1_start) * (d2_end - d2_start);
	return layout_from_fields(0, row_elems * elem, d1_stride, d0_start * elem, d1_start, d2_start, d0_end * elem, d1_end, d2_end, 0, total);
}

// a random window of a random box, and a random target box with a window of the same length
std::pair<data_layout, data_layout> random_layouts(std::mt19937_64& rng) {
	auto source = random_box(rng);
	const int64_t source_total = source.end;
	source.start = pick(rng, 0, source_total - 1);
	source.end = pick(rng, source.start + 1, source_total);
	const int64_t length = source.end - source.start;

	data_layout target = normal_form::one_run(0, pick(rng, 0, 64), length);
	for(int attempt = 0; attempt < 20; ++attempt) {
		auto candidate = random_box(rng);
		if(candidate.end < length) continue;
		candidate.start = pick(rng, 0, candidate.end - length);
		candidate.end = candidate.start + length;
		target = candidate;
		break;
	}
	return {source, target};
}

} // namespace

TEST_CASE("randomized layouts, windows, ends and strategies", "[backend][random]") {
	// set COPYLIB_TEST_SEED to reproduce a reported case; cases are generated in sequence, so the index identifies it
	const char* seed_override = std::getenv("COPYLIB_TEST_SEED");
	const uint64_t seed = seed_override != nullptr ? std::strtoull(seed_override, nullptr, 10) : 20260925;
	CAPTURE(seed);
	std::mt19937_64 rng(seed);

	constexpr int64_t chunk_sizes[] = {0, 1, 8, 40, 64};
	auto exec = make_executor();
	copy_report report;
	const auto pairs = endpoint_pairs();
	for(int i = 0; i < 60; ++i) {
		const auto [source, target] = random_layouts(rng);
		const auto& [from, to] = pairs[static_cast<size_t>(pick(rng, 0, static_cast<int64_t>(pairs.size()) - 1))];
		const auto strategy = strategy_from_fields(pick(rng, 0, 1) == 0 ? copy_type::direct : copy_type::staged,
		    pick(rng, 0, 1) == 0 ? copy_properties::none : copy_properties::use_kernel, static_cast<d2d_implementation>(pick(rng, 0, 3)),
		    chunk_sizes[pick(rng, 0, 4)]);

		const prepared_copy copy(exec, from, source, to, target);
		report.add(run_copy(exec, copy, strategy), "case " + std::to_string(i) + ": " + utils::format("{}", copy.spec()) + " " + where(from, to, strategy));
	}
	report.check();
}
