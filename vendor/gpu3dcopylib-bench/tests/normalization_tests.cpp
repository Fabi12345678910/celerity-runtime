#include "test_utils.hpp"

#include <copylib/core.hpp>

#include <catch2/catch_test_macros.hpp>

// Layer 3: normalize() maps a layout to the unique normal form of its box (docs/design.md, "Normal form and equality").
//
// The first group of cases pins what normalization means independently of the encoding: the bytes are preserved and
// each row of the normalized box is one maximal contiguous run. The second group pins the exact encoding, which is what
// makes comparing normalized layouts field by field equivalent to comparing the bytes they describe.

using namespace copylib;
using namespace copylib_testing;
using namespace copylib_testing::normal_form;
namespace ref = copylib_testing::reference_box;

namespace {

// the layouts every property is checked on, with full windows
std::vector<data_layout> full_window_layouts() {
	return {ref::fields(), shapes::whole_allocation(), shapes::single_row(), shapes::full_rows_of_one_plane(), shapes::full_rows_of_three_planes(),
	    shapes::partial_rows_of_two_full_planes()};
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// Semantics

TEST_CASE("normalization preserves the bytes a layout describes", "[normalization]") {
	auto layouts = full_window_layouts();
	layouts.push_back(with_window_fields(shapes::whole_allocation(), 100, 5000));
	layouts.push_back(with_window_fields(shapes::full_rows_of_three_planes(), 300, 700));

	for(const auto& layout : layouts) {
		const auto normalized = normalize(layout);
		CHECK(space_of(normalized) == space_of(layout));
		CHECK(first_difference(reference_offsets(normalized), reference_offsets(layout)) == -1);
	}
}

TEST_CASE("normalization leaves the window unchanged", "[normalization][window]") {
	// collapsing keeps the packed order, which is ascending address order, so packed offsets stay the same
	const auto layout = with_window_fields(shapes::whole_allocation(), 100, 5000);
	const auto normalized = normalize(layout);

	CHECK(normalized.start == 100);
	CHECK(normalized.end == 5000);
}

TEST_CASE("each row of a normalized box is a maximal contiguous run", "[normalization]") {
	for(const auto& layout : full_window_layouts()) {
		const auto runs = reference_runs(layout);
		const auto normalized = normalize(layout);

		CHECK(rows_in_box(normalized) == static_cast<int64_t>(runs.size()));
		CHECK(row_extent_of(normalized) == runs.front().length);
	}
}

TEST_CASE("normalization is idempotent", "[normalization]") {
	for(const auto& layout : full_window_layouts()) {
		const auto once = normalize(layout);
		// without this, a normalize returning an empty layout would pass: normalize({}) == {}
		REQUIRE(first_difference(reference_offsets(once), reference_offsets(layout)) == -1);
		CHECK(same_fields(normalize(once), once));
	}
}

TEST_CASE("a normalized layout is valid", "[normalization]") {
	for(const auto& layout : full_window_layouts()) {
		CHECK(is_valid(normalize(layout)));
	}
}

// ---------------------------------------------------------------------------------------------------------------------
// The normal form

TEST_CASE("a single run normalizes to the 1D form", "[normalization]") {
	// both strides become the end of the run, regardless of the allocation the run came from
	CHECK(same_fields(normalize(shapes::whole_allocation()), one_run(ref::base, 0, 12 * ref::plane_bytes)));
	CHECK(same_fields(normalize(shapes::full_rows_of_one_plane()), one_run(ref::base, 240, 320)));
	CHECK(same_fields(normalize(shapes::single_row()), one_run(ref::base, 16, 24)));
}

TEST_CASE("evenly spaced runs normalize to the 2D form", "[normalization]") {
	SECTION("one run per plane") { CHECK(same_fields(normalize(shapes::full_rows_of_three_planes()), uniform_runs(ref::base, 2800, 320, 3, 1280))); }

	SECTION("partial rows continuing across a plane boundary") {
		// no two rows are adjacent, but a plane is exactly 16 rows, so all 32 rows are spaced 80 bytes apart
		CHECK(same_fields(normalize(shapes::partial_rows_of_two_full_planes()), uniform_runs(ref::base, 16, 24, 32, 80)));
	}
}

TEST_CASE("a two-level grid is already in normal form", "[normalization]") {
	CHECK(same_fields(normalize(ref::fields()), ref::fields()));
	// paired with a collapsing case, so that the case cannot pass while normalize() returns its input unchanged
	CHECK(same_fields(normalize(shapes::single_row()), one_run(ref::base, 16, 24)));
}

TEST_CASE("the normal form is unique", "[normalization]") {
	// different encodings of the same bytes normalize to identical fields

	SECTION("a single row, encoded with three different sets of strides") {
		const auto in_allocation = shapes::single_row();                                    // 80 byte rows, 16 per plane
		const auto tight = layout_from_fields(ref::base, 40, 2, 16, 0, 0, 40, 1, 1, 0, 24); // 40 byte rows, 2 per plane
		const auto already_normal = one_run(ref::base, 16, 24);                             // the normal form itself

		CHECK(same_fields(normalize(in_allocation), normalize(tight)));
		CHECK(same_fields(normalize(tight), normalize(already_normal)));
		// comparing two normalized results alone would pass for a normalize returning {} both times
		CHECK(same_fields(normalize(in_allocation), already_normal));
	}

	SECTION("one run per plane, as planes or as rows") {
		const auto as_rows = uniform_runs(ref::base, 2800, 320, 3, 1280);
		CHECK(same_fields(normalize(shapes::full_rows_of_three_planes()), normalize(as_rows)));
		CHECK(same_fields(normalize(shapes::full_rows_of_three_planes()), as_rows));
	}
}

// ---------------------------------------------------------------------------------------------------------------------
// Specs

TEST_CASE("normalizing a spec normalizes each side independently", "[normalization]") {
	// the source is one contiguous run of 320 bytes, the target eight runs of 40 bytes; under the 2D library's rule of
	// collapsing only as far as both sides allow, the source would stay uncollapsed
	const auto source = shapes::full_rows_of_one_plane(0x10000);
	const auto target = uniform_runs(0x80000, 0, 40, 8, 80);
	const auto spec = spec_from_fields(device_id::d0, source, device_id::d1, target);

	const auto normalized = normalize(spec);

	CHECK(same_fields(normalized.source_layout, one_run(0x10000, 240, 320)));
	CHECK(same_fields(normalized.target_layout, target));
	CHECK(expected_mapping(normalized) == expected_mapping(spec));
	CHECK(normalized.source_device == spec.source_device);
	CHECK(normalized.target_device == spec.target_device);
}

TEST_CASE("normalizing a spec preserves what it copies", "[normalization]") {
	// 768 bytes from the front of a contiguous allocation into 32 partial rows: both sides collapse, to different forms
	constexpr int64_t length = 768;
	const auto source = with_window_fields(shapes::whole_allocation(0x10000), 0, length);
	const auto target = shapes::partial_rows_of_two_full_planes(0x80000);
	const auto spec = spec_from_fields(device_id::d0, source, device_id::d1, target);
	REQUIRE(reference_offsets(target).size() == static_cast<size_t>(length));

	const auto normalized = normalize(spec);

	CHECK(expected_mapping(normalized) == expected_mapping(spec));
	CHECK(same_fields(normalized.source_layout, with_window_fields(one_run(0x10000, 0, 12 * ref::plane_bytes), 0, length)));
	CHECK(same_fields(normalized.target_layout, uniform_runs(0x80000, 16, 24, 32, 80)));
}
