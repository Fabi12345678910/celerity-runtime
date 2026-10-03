#include "test_utils.hpp"

#include <copylib/core.hpp>

#include <catch2/catch_test_macros.hpp>

// Tests for the test oracle in test_utils.hpp. Every planning test relies on reference_offsets() and simulate(), so
// they are checked here against hand-computed values and hand-built plans. These use no library functions beyond
// plain field access, and are expected to pass from the start.

using namespace copylib;
using namespace copylib_testing;
namespace ref = copylib_testing::reference_box;

TEST_CASE("the reference interpreter agrees with the documented example", "[reference]") {
	const auto offsets = reference_offsets(ref::fields());

	REQUIRE(offsets.size() == static_cast<size_t>(ref::total_bytes));
	CHECK(offsets.front() == ref::first_byte);           // 2816
	CHECK(offsets[ref::row_extent] == 2896);             // next row
	CHECK(offsets[ref::row_extent * ref::rows] == 4096); // next plane
	CHECK(offsets.back() == 5639);                       // last byte of the box
}

TEST_CASE("the reference interpreter respects the window", "[reference]") {
	const auto offsets = reference_offsets(with_window_fields(ref::fields(), 12, 40));

	REQUIRE(offsets.size() == 28);
	CHECK(offsets.front() == 2828);
	CHECK(offsets[11] == 2839); // last byte of the first row
	CHECK(offsets[12] == 2896); // first byte of the second row
}

TEST_CASE("the reference interpreter rejects malformed layouts", "[reference]") {
	CHECK(reference_offsets(data_layout{}).empty());
	CHECK(reference_offsets(with_window_fields(ref::fields(), 10, 10)).empty());
	CHECK(reference_offsets(with_window_fields(ref::fields(), 0, ref::total_bytes + 1)).empty());
}

TEST_CASE("reference runs are maximal", "[reference]") {
	CHECK(reference_runs(ref::fields()).size() == 12); // rows are 24 of 80 bytes, so none are adjacent
	CHECK(reference_runs(shapes::whole_allocation()).size() == 1);
	CHECK(reference_runs(shapes::full_rows_of_one_plane()).size() == 1);
	CHECK(reference_runs(shapes::full_rows_of_three_planes()).size() == 3);
	CHECK(reference_runs(shapes::single_row()).size() == 1);
}

TEST_CASE("the normal form builders describe the bytes of the shapes they stand for", "[reference]") {
	using namespace copylib_testing::normal_form;
	constexpr intptr_t at = ref::base;

	CHECK(first_difference(reference_offsets(one_run(at, 0, 12 * ref::plane_bytes)), reference_offsets(shapes::whole_allocation())) == -1);
	CHECK(first_difference(reference_offsets(one_run(at, 16, 24)), reference_offsets(shapes::single_row())) == -1);
	CHECK(first_difference(reference_offsets(one_run(at, 240, 320)), reference_offsets(shapes::full_rows_of_one_plane())) == -1);
	CHECK(first_difference(reference_offsets(uniform_runs(at, 2800, 320, 3, 1280)), reference_offsets(shapes::full_rows_of_three_planes())) == -1);
	CHECK(first_difference(reference_offsets(uniform_runs(at, 16, 24, 32, 80)), reference_offsets(shapes::partial_rows_of_two_full_planes())) == -1);
}

TEST_CASE("a direct spec implements itself", "[reference]") {
	const auto spec = ref::spec();

	CHECK(implements(copy_plan{spec}, spec));
	CHECK(implements(parallel_copy_set{copy_plan{spec}}, spec));
}

TEST_CASE("the simulation follows values through staging buffers", "[reference]") {
	const auto spec = ref::spec();
	const auto staging = staging_layout_from_fields(staging_id{}, ref::total_bytes); // default-constructed id is flagged as staging

	const auto gather = spec_from_fields(device_id::d0, spec.source_layout, device_id::d0, staging);
	const auto scatter = spec_from_fields(device_id::d0, staging, device_id::d1, spec.target_layout);

	CHECK(implements(copy_plan{gather, scatter}, spec));
	// the other way round, the scatter reads a staging buffer nobody wrote
	CHECK_FALSE(implements(copy_plan{scatter, gather}, spec));
}

TEST_CASE("the simulation detects gaps, overlaps and stray writes", "[reference]") {
	const auto spec = ref::spec();
	const auto chunk = [&](int64_t start, int64_t end) {
		return copy_plan{spec_from_fields(
		    device_id::d0, with_window_fields(spec.source_layout, start, end), device_id::d1, with_window_fields(spec.target_layout, start, end))};
	};

	CHECK(implements(parallel_copy_set{chunk(0, 96), chunk(96, 288)}, spec));
	CHECK(implements(parallel_copy_set{chunk(96, 288), chunk(0, 96)}, spec)); // order within a set is irrelevant

	CHECK_FALSE(implements(parallel_copy_set{chunk(0, 96), chunk(120, 288)}, spec)); // gap
	CHECK_FALSE(implements(parallel_copy_set{chunk(0, 120), chunk(96, 288)}, spec)); // overlap
	CHECK_FALSE(implements(parallel_copy_set{chunk(0, 288), chunk(0, 24)}, spec));   // written twice
	CHECK_FALSE(implements(parallel_copy_set{}, spec));

	// right bytes, wrong place: target window shifted by one row
	const auto shifted = copy_plan{
	    spec_from_fields(device_id::d0, with_window_fields(spec.source_layout, 0, 264), device_id::d1, with_window_fields(spec.target_layout, 24, 288))};
	CHECK_FALSE(implements(shifted, spec));
}

TEST_CASE("the simulation detects plans sharing a staging buffer", "[reference]") {
	const auto spec = ref::spec();
	const auto staging = staging_layout_from_fields(staging_id{}, 144);
	const auto half = [&](int64_t start) {
		const auto source = with_window_fields(spec.source_layout, start, start + 144);
		const auto target = with_window_fields(spec.target_layout, start, start + 144);
		return copy_plan{spec_from_fields(device_id::d0, source, device_id::d0, staging), spec_from_fields(device_id::d0, staging, device_id::d1, target)};
	};

	// each half on its own is fine, but run concurrently they overwrite each other's staging buffer
	CHECK(simulate(half(0)).consistent);
	CHECK(simulate(parallel_copy_set{half(0), half(144)}).conflicting);
}
