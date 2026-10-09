#include "test_utils.hpp"

#include <copylib/core.hpp>

#include <catch2/catch_test_macros.hpp>

// Layer 2: the validity rules for layouts, specs, plans and sets, as listed in docs/design.md.
//
// Every case pairs the rejection of a malformed input with the acceptance of a well-formed one. That matters while
// `is_valid` is incomplete: without the positive check, a rejection test would pass for the wrong reason and report
// progress that is not there.

using namespace copylib;
using copylib_testing::layout_from_fields;
using copylib_testing::spec_from_fields;
using copylib_testing::with_window_fields;
namespace ref = copylib_testing::reference_box;
namespace shapes = copylib_testing::shapes;

TEST_CASE("a well-formed layout is valid", "[validation]") { CHECK(is_valid(ref::fields())); }

TEST_CASE("strides must be positive", "[validation]") {
	CHECK(is_valid(ref::fields()));

	auto no_d0 = ref::fields();
	no_d0.d0_stride = 0;
	CHECK_FALSE(is_valid(no_d0));

	auto no_d1 = ref::fields();
	no_d1.d1_stride = 0;
	CHECK_FALSE(is_valid(no_d1));
}

TEST_CASE("the box must lie inside one row and one plane", "[validation]") {
	// this is also what makes the encoding canonical: a box that could be expressed by moving whole rows or planes
	// into the next dimension has more than one encoding, which would break operator== and hashing
	CHECK(is_valid(ref::fields()));

	SECTION("a row may not be longer than the row stride") {
		auto layout = ref::fields();
		layout.d0_end_offset = ref::d0_stride + ref::elem_size;
		CHECK_FALSE(is_valid(layout));
	}

	SECTION("a row may not start past the row stride") {
		auto layout = ref::fields();
		layout.d0_start_offset = ref::d0_stride;
		layout.d0_end_offset = ref::d0_stride + ref::row_extent;
		CHECK_FALSE(is_valid(layout));
	}

	SECTION("a box may not hold more rows than a plane has") {
		auto layout = ref::fields();
		layout.d1_end_offset = ref::d1_stride + 1;
		CHECK_FALSE(is_valid(layout));
	}

	SECTION("a box may not start past the last row of a plane") {
		auto layout = ref::fields();
		layout.d1_start_offset = ref::d1_stride;
		layout.d1_end_offset = ref::d1_stride + 1;
		CHECK_FALSE(is_valid(layout));
	}
}

TEST_CASE("every dimension must be non-empty", "[validation]") {
	CHECK(is_valid(ref::fields()));

	for(const int dimension : {0, 1, 2}) {
		CAPTURE(dimension);
		auto layout = ref::fields();
		if(dimension == 0) { layout.d0_end_offset = layout.d0_start_offset; }
		if(dimension == 1) { layout.d1_end_offset = layout.d1_start_offset; }
		if(dimension == 2) { layout.d2_end_offset = layout.d2_start_offset; }
		CHECK_FALSE(is_valid(layout));
	}
}

TEST_CASE("the plane count of the allocation is not bounded", "[validation]") {
	// there is no d2_stride, so nothing says how many planes the allocation has
	auto layout = ref::fields();
	layout.d2_start_offset = 1000;
	layout.d2_end_offset = 1003;
	CHECK(is_valid(layout));

	layout.d2_start_offset = -1;
	CHECK_FALSE(is_valid(layout));
}

TEST_CASE("the window must lie within the box", "[validation][window]") {
	CHECK(is_valid(ref::fields()));

	SECTION("a window may not extend past the end of the box") {
		auto layout = ref::fields();
		layout.end = ref::total_bytes + 1;
		CHECK_FALSE(is_valid(layout));
	}

	SECTION("a window may not be empty") {
		auto layout = ref::fields();
		layout.start = layout.end;
		CHECK_FALSE(is_valid(layout));
	}

	SECTION("a window may not start before the box") {
		auto layout = ref::fields();
		layout.start = -1;
		CHECK_FALSE(is_valid(layout));
	}

	SECTION("a sub-window of the box is valid") { CHECK(is_valid(with_window_fields(ref::fields(), ref::row_extent, 2 * ref::row_extent))); }
}

TEST_CASE("a copy spec requires equal window lengths on both sides", "[validation]") {
	const auto source = ref::fields();

	SECTION("equal windows are valid") { CHECK(is_valid(spec_from_fields(device_id::d0, source, device_id::d1, ref::fields(0x80000)))); }

	SECTION("differing window lengths are not") {
		auto target = ref::fields(0x80000);
		target.end = ref::total_bytes - ref::elem_size;
		CHECK_FALSE(is_valid(spec_from_fields(device_id::d0, source, device_id::d1, target)));
	}

	SECTION("boxes of different sizes are fine as long as the windows match") {
		// a staged chunk: half of the box gathered into a buffer that holds exactly those bytes
		const auto half = with_window_fields(source, 0, ref::total_bytes / 2);
		const auto buffer = copylib_testing::staging_layout_from_fields(copylib_testing::staging_id_from_fields(false, device_id::d0, 0), ref::total_bytes / 2);
		CHECK(is_valid(spec_from_fields(device_id::d0, half, device_id::d0, buffer)));
	}
}

TEST_CASE("a reshaping copy spec is valid", "[validation]") {
	// design.md: source and target may have different shapes, so int[1,1,6] -> int[1,2,3] is allowed
	const auto source = shapes::one_row_of_six(0x10000);
	const auto target = shapes::two_rows_of_three(0x20000);

	CHECK(is_valid(source));
	CHECK(is_valid(target));
	CHECK(is_valid(spec_from_fields(device_id::d0, source, device_id::d1, target)));
}

TEST_CASE("an empty copy plan or set is valid", "[validation]") {
	// an empty plan copies nothing, which is not an error
	CHECK(is_valid(copy_plan{}));
	CHECK(is_valid(parallel_copy_set{}));

	// paired with rejections, so that the case cannot pass for an is_valid that accepts everything
	auto broken = ref::spec();
	broken.target_layout.end -= 8;
	CHECK_FALSE(is_valid(copy_plan{ref::spec(), broken}));
	CHECK_FALSE(is_valid(parallel_copy_set{copy_plan{ref::spec()}, copy_plan{broken}}));
}
