#include "test_utils.hpp"

#include <copylib/core.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

// Layer 1: the pure layout math, no SYCL involved.
//
// Only the constructor cases build layouts through a constructor; everything else assigns the fields directly via
// layout_from_fields(), so that a failure points at one layer instead of cascading from the constructors.

using namespace copylib;
using copylib_testing::layout_from_fields;
using copylib_testing::reference_offsets;
using copylib_testing::spec_from_fields;
using copylib_testing::with_window_fields;
using copylib_testing::shapes::full_rows_of_one_plane;
using copylib_testing::shapes::one_row_of_six;
using copylib_testing::shapes::single_row;
using copylib_testing::shapes::six_rows_of_48;
using copylib_testing::shapes::two_rows_of_three;
using copylib_testing::shapes::whole_allocation;
namespace ref = copylib_testing::reference_box;

namespace {

struct copy_run {
	int64_t source_offset;
	int64_t target_offset;
	int64_t length;
};

std::vector<copy_run> collect_copy_runs(const copy_spec& spec) {
	std::vector<copy_run> runs;
	for_each_copy_run(spec, [&](int64_t source_offset, int64_t target_offset, int64_t length) { runs.push_back({source_offset, target_offset, length}); });
	return runs;
}

// same shape, strided to contiguous, strided reshape, contiguous reshape, and windows starting mid-row on both sides
std::vector<copy_spec> copy_run_cases() {
	return {
	    ref::spec(),
	    spec_from_fields(device_id::d0, ref::fields(), device_id::d1, copylib_testing::normal_form::one_run(0x80000, 0, ref::total_bytes)),
	    spec_from_fields(device_id::d0, six_rows_of_48(0x10000), device_id::d1, ref::fields(0x80000)),
	    spec_from_fields(device_id::d0, one_row_of_six(0x10000), device_id::d1, two_rows_of_three(0x80000)),
	    spec_from_fields(device_id::d0, with_window_fields(ref::fields(), 7, 283), device_id::d1, with_window_fields(whole_allocation(0x80000), 100, 376)),
	};
}

} // namespace

TEST_CASE("the box constructor stores the layout fields", "[layout]") {
	const auto layout = ref::make();

	CHECK(layout.base == ref::base);
	CHECK(layout.d0_stride == ref::d0_stride);
	CHECK(layout.d1_stride == ref::d1_stride);
	CHECK(layout.d0_start_offset == ref::d0_start_offset);
	CHECK(layout.d1_start_offset == ref::d1_start_offset);
	CHECK(layout.d2_start_offset == ref::d2_start_offset);
	CHECK(layout.d0_end_offset == ref::d0_end_offset);
	CHECK(layout.d1_end_offset == ref::d1_end_offset);
	CHECK(layout.d2_end_offset == ref::d2_end_offset);
}

TEST_CASE("a freshly constructed layout has a window covering the whole box", "[layout][window]") {
	// design.md: "Constructors set the full box explicitly. There is no '0 means everything' shorthand."
	const auto layout = ref::make();

	CHECK(layout.start == 0);
	CHECK(layout.end == ref::total_bytes);
}

TEST_CASE("a 1D constructor describes a contiguous run", "[layout]") {
	constexpr int64_t offset = 128;
	constexpr int64_t length = 256;
	const data_layout layout{ref::base, offset, length};

	CHECK(layout.total_bytes() == length);
	CHECK(layout.window_length() == length);
	CHECK(layout.is_window_contiguous());
	CHECK(layout.offset_at(0) == offset);
	// design.md: the 1D constructor produces exactly the 1D normal form
	CHECK(copylib_testing::same_fields(layout, copylib_testing::normal_form::one_run(ref::base, offset, length)));
}

TEST_CASE("layouts compare equal exactly when all their fields match", "[layout][equality]") {
	const auto layout = ref::fields();

	CHECK(layout == ref::fields());
	CHECK_FALSE(layout != ref::fields());

	SECTION("a different window") {
		const auto other = with_window_fields(layout, 0, 24);
		CHECK(layout != other);
		CHECK_FALSE(layout == other);
	}

	SECTION("a different allocation") { CHECK(layout != ref::fields(ref::base + 0x10000)); }

	SECTION("the same bytes with different strides") {
		// equality compares encodings, not bytes
		const auto in_allocation = single_row();
		const auto normal = copylib_testing::normal_form::one_run(ref::base, 16, 24);
		CHECK(in_allocation != normal);
		CHECK(normalize(in_allocation) == normalize(normal));
	}
}

TEST_CASE("the window length is the size of the copy", "[layout][window]") {
	auto layout = ref::fields();
	CHECK(layout.window_length() == ref::total_bytes);

	layout.start = 24;
	layout.end = 48;
	CHECK(layout.window_length() == 24);
}

TEST_CASE("the reference box covers the documented number of bytes", "[layout]") {
	// design.md: 24 bytes x 4 rows x 3 planes = 288 bytes, starting at byte 2816
	CHECK(ref::total_bytes == 288); // guards the test's own arithmetic
	CHECK(ref::first_byte == 2816); // ditto

	CHECK(ref::fields().total_bytes() == 288);
}

TEST_CASE("offset_at maps packed offsets into the allocation", "[layout]") {
	const auto layout = ref::fields();

	CHECK(layout.offset_at(0) == ref::first_byte);                // 2816, first byte of the box
	CHECK(layout.offset_at(ref::row_extent - 1) == 2839);         // last byte of the first row
	CHECK(layout.offset_at(ref::row_extent) == 2896);             // first byte of the second row, one row stride on
	CHECK(layout.offset_at(ref::row_extent * ref::rows) == 4096); // first byte of the second plane
	CHECK(layout.offset_at(ref::total_bytes - 1) == 5639);        // last byte of the box
}

TEST_CASE("a strided box is not contiguous", "[layout]") {
	// the box covers 24 of the 80 bytes of each row, and 4 of the 16 rows of each plane
	const auto layout = ref::fields();

	CHECK_FALSE(layout.is_window_contiguous());
	CHECK_FALSE(layout.d1_contiguous());
	CHECK_FALSE(layout.d2_contiguous());

	// paired with a positive, so that predicates always returning false cannot pass
	CHECK(whole_allocation().is_window_contiguous());
}

TEST_CASE("a box spanning the whole allocation is contiguous", "[layout]") {
	const auto layout = whole_allocation();

	CHECK(layout.total_bytes() == 12 * ref::plane_bytes);
	CHECK(layout.d1_contiguous());
	CHECK(layout.d2_contiguous());
	CHECK(layout.is_window_contiguous());
}

TEST_CASE("a box covering a single row is contiguous", "[layout]") {
	const auto layout = single_row();

	CHECK(layout.total_bytes() == ref::row_extent);
	CHECK(layout.is_window_contiguous());
}

TEST_CASE("full rows of one plane have collapsible fragments", "[layout]") {
	const auto layout = full_rows_of_one_plane();

	CHECK(layout.total_bytes() == ref::rows * ref::d0_stride);
	CHECK(layout.d1_contiguous());
	CHECK(layout.is_window_contiguous());
}

TEST_CASE("contiguity is a property of the window, not of the box", "[layout][window]") {
	// design.md: "A window inside one row is a single queue.copy."
	auto layout = ref::fields();
	REQUIRE_FALSE(layout.is_window_contiguous());

	layout.start = 2;
	layout.end = 10;
	CHECK(layout.is_window_contiguous());
}

TEST_CASE("offset_at agrees with the reference interpreter for every byte", "[layout]") {
	for(const auto& layout : {ref::fields(), whole_allocation(), single_row(), full_rows_of_one_plane(), with_window_fields(ref::fields(), 7, 283)}) {
		const auto expected = reference_offsets(layout);
		REQUIRE_FALSE(expected.empty());
		bool all_match = true;
		for(int64_t i = 0; i < static_cast<int64_t>(expected.size()); ++i) {
			all_match = all_match && layout.offset_at(layout.start + i) == expected[static_cast<size_t>(i)];
		}
		CHECK(all_match);
	}
}

TEST_CASE("the copy runs pair every byte with its reference offset on both sides", "[layout][window]") {
	for(const auto& spec : copy_run_cases()) {
		CAPTURE(spec);
		std::vector<int64_t> source_covered;
		std::vector<int64_t> target_covered;
		for(const auto& run : collect_copy_runs(spec)) {
			CHECK(run.length > 0);
			for(int64_t i = 0; i < run.length; ++i) {
				source_covered.push_back(run.source_offset + i);
				target_covered.push_back(run.target_offset + i);
			}
		}
		CHECK(copylib_testing::first_difference(source_covered, reference_offsets(spec.source_layout)) == -1);
		CHECK(copylib_testing::first_difference(target_covered, reference_offsets(spec.target_layout)) == -1);
	}
}

TEST_CASE("consecutive copy runs are split only where a side is not contiguous", "[layout][window]") {
	for(const auto& spec : copy_run_cases()) {
		CAPTURE(spec);
		const auto runs = collect_copy_runs(spec);
		for(size_t r = 1; r < runs.size(); ++r) {
			const auto& previous = runs[r - 1];
			CHECK((previous.source_offset + previous.length != runs[r].source_offset || previous.target_offset + previous.length != runs[r].target_offset));
		}
	}
}

TEST_CASE("a copy has as many runs as its more fragmented side", "[layout][window]") {
	// contiguous on both sides, across rows and planes
	CHECK(collect_copy_runs(spec_from_fields(device_id::d0, whole_allocation(), device_id::d1, whole_allocation(0x80000))).size() == 1);
	CHECK(collect_copy_runs(spec_from_fields(device_id::d0, one_row_of_six(0x10000), device_id::d1, two_rows_of_three(0x80000))).size() == 1);
	// the reference box has 12 separate rows of 24 bytes, whatever the other side looks like
	CHECK(collect_copy_runs(ref::spec()).size() == 12);
	CHECK(collect_copy_runs(spec_from_fields(device_id::d0, six_rows_of_48(0x10000), device_id::d1, ref::fields(0x80000))).size() == 12);
}

TEST_CASE("the base pointer of a placed layout is its base address", "[layout]") {
	const auto layout = ref::fields();

	CHECK(layout.base_ptr() == reinterpret_cast<std::byte*>(ref::base));
	CHECK_FALSE(layout.is_unplaced_staging());
}

TEST_CASE("a staging layout is unplaced until it is fulfilled", "[layout][staging]") {
	auto layout = ref::fields();
	layout.staging = staging_id{false, device_id::d1, 7};

	CHECK(layout.is_unplaced_staging());
}

TEST_CASE("staging ids round-trip their fields", "[layout][staging]") {
	const staging_id staging{true, device_id::d3, 42};

	CHECK(staging.on_host == true);
	CHECK(staging.did == device_id::d3);
	CHECK(staging.index == 42);
	CHECK(staging.is_staging_id == staging_id::staging_id_flag);
}

TEST_CASE("staging ids are distinguishable from pointers", "[layout][staging]") {
	// This holds by construction (the flag byte is a default member initializer) and is what makes
	// is_unplaced_staging() implementable, so it is expected to pass already.
	const staging_id staging{};

	CHECK(staging.is_staging_id == staging_id::staging_id_flag);
	CHECK(sizeof(staging_id) == sizeof(intptr_t));
	// a layout base must be at least 2-byte aligned so that it can never look like a staging id
	CHECK((ref::base & 0x1) == 0);
}
