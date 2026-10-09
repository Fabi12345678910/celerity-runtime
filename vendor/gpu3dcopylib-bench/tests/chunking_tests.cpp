#include "test_utils.hpp"

#include <copylib/core.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <vector>

// Layer 4: apply_chunking() splits a spec into independent single-spec plans of at most chunk_size bytes each.
//
// design.md: chunking keeps the box and narrows only the window; chunk boundaries are rounded to copy_alignment(), the
// largest power of two up to 64 dividing both sides' row extents, d0_stride and d0_start_offset and the shift between
// the windows; chunk_size == 0 means no chunking.

using namespace copylib;
using namespace copylib_testing;
namespace ref = copylib_testing::reference_box;

namespace {

// a set's plans are unordered, so compare chunks by where their window starts
std::vector<copy_spec> chunks_by_start(const parallel_copy_set& set) {
	std::vector<copy_spec> chunks;
	for(const auto& plan : set) {
		for(const auto& spec : plan) {
			chunks.push_back(spec);
		}
	}
	std::sort(chunks.begin(), chunks.end(), [](const copy_spec& a, const copy_spec& b) { return a.source_layout.start < b.source_layout.start; });
	return chunks;
}

std::vector<int64_t> chunk_lengths(const parallel_copy_set& set) {
	std::vector<int64_t> lengths;
	for(const auto& chunk : chunks_by_start(set)) {
		lengths.push_back(chunk.source_layout.window_length());
	}
	return lengths;
}

copy_spec reshaping_spec() { return spec_from_fields(device_id::d0, shapes::one_row_of_six(0x10000), device_id::d1, shapes::two_rows_of_three(0x20000)); }

} // namespace

TEST_CASE("a chunk size of zero means no chunking", "[chunking]") {
	const auto spec = ref::spec();
	const auto set = apply_chunking(spec, chunking_strategy(0));

	REQUIRE(set.size() == 1);
	REQUIRE(set.front().size() == 1);
	const auto& only = set.front().front();
	CHECK(same_fields(only.source_layout, spec.source_layout));
	CHECK(same_fields(only.target_layout, spec.target_layout));
	CHECK(only.source_device == spec.source_device);
	CHECK(only.target_device == spec.target_device);
}

TEST_CASE("chunks implement the spec", "[chunking]") {
	const int64_t chunk_size = GENERATE(8, 24, 64, 77, 96, 100, 288, 1000);
	CAPTURE(chunk_size);

	const auto spec = ref::spec();
	const auto set = apply_chunking(spec, chunking_strategy(chunk_size));

	CHECK(implements(set, spec));
}

TEST_CASE("every chunk is a single direct copy of at most chunk_size bytes", "[chunking]") {
	const int64_t chunk_size = GENERATE(8, 24, 64, 77, 96, 100);
	CAPTURE(chunk_size);

	const auto spec = ref::spec();
	const auto set = apply_chunking(spec, chunking_strategy(chunk_size));

	REQUIRE_FALSE(set.empty());
	for(const auto& plan : set) {
		REQUIRE(plan.size() == 1);
		CHECK(plan.front().source_layout.window_length() <= chunk_size);
		CHECK(plan.front().target_layout.window_length() == plan.front().source_layout.window_length());
	}
}

TEST_CASE("chunking keeps the box and narrows only the window", "[chunking][window]") {
	const auto spec = ref::spec();
	const auto set = apply_chunking(spec, chunking_strategy(64));

	REQUIRE_FALSE(set.empty());
	for(const auto& chunk : chunks_by_start(set)) {
		CHECK(same_box(chunk.source_layout, spec.source_layout));
		CHECK(same_box(chunk.target_layout, spec.target_layout));
		CHECK(chunk.source_device == spec.source_device);
		CHECK(chunk.target_device == spec.target_device);
	}
}

TEST_CASE("source and target windows advance in lockstep", "[chunking][window]") {
	const auto spec = GENERATE(ref::spec(), reshaping_spec());
	const auto set = apply_chunking(spec, chunking_strategy(8));

	REQUIRE(set.size() > 1);
	for(const auto& chunk : chunks_by_start(set)) {
		CHECK(chunk.source_layout.start - spec.source_layout.start == chunk.target_layout.start - spec.target_layout.start);
		CHECK(chunk.source_layout.window_length() == chunk.target_layout.window_length());
	}
}

TEST_CASE("chunk boundaries are aligned", "[chunking]") {
	// the reference box has row extent 24, d0_stride 80 and d0_start_offset 16, so its alignment is 8
	const int64_t chunk_size = GENERATE(64, 77, 100);
	CAPTURE(chunk_size);

	const auto set = apply_chunking(ref::spec(), chunking_strategy(chunk_size));

	REQUIRE_FALSE(set.empty());
	for(const auto& chunk : chunks_by_start(set)) {
		CHECK(chunk.source_layout.start % ref::alignment == 0);
		CHECK(chunk.source_layout.end % ref::alignment == 0);
	}
}

TEST_CASE("chunks are as large as alignment permits", "[chunking]") {
	SECTION("perfectly divisible") { CHECK(chunk_lengths(apply_chunking(ref::spec(), chunking_strategy(96))) == std::vector<int64_t>{96, 96, 96}); }

	SECTION("with a remainder, which goes into the last chunk") {
		CHECK(chunk_lengths(apply_chunking(ref::spec(), chunking_strategy(64))) == std::vector<int64_t>{64, 64, 64, 64, 32});
	}

	SECTION("an unaligned chunk size is rounded down to the alignment") {
		// each chunk may be at most 77 bytes, and the largest multiple of 8 below that is 72
		CHECK(chunk_lengths(apply_chunking(ref::spec(), chunking_strategy(77))) == std::vector<int64_t>{72, 72, 72, 72});
	}

	SECTION("a chunk size larger than the window yields a single chunk") {
		CHECK(chunk_lengths(apply_chunking(ref::spec(), chunking_strategy(1000))) == std::vector<int64_t>{288});
	}
}

TEST_CASE("a chunk size below the alignment is raised to it", "[chunking]") {
	// design.md: the reference box has alignment 8, so chunks of 4 or 1 byte become chunks of 8
	const int64_t chunk_size = GENERATE(1, 4, 7);
	CAPTURE(chunk_size);

	const auto spec = ref::spec();
	const auto set = apply_chunking(spec, chunking_strategy(chunk_size));

	CHECK(implements(set, spec));
	CHECK(chunk_lengths(set) == std::vector<int64_t>(ref::total_bytes / ref::alignment, ref::alignment));
}

TEST_CASE("chunk boundaries of an unaligned window sit at absolute aligned offsets", "[chunking][window]") {
	// design.md: boundaries are multiples of the alignment in packed offsets, not relative to the window start, so only
	// the window's own ends can be unaligned
	auto spec = ref::spec();
	const auto window = [&](int64_t start, int64_t end) {
		spec.source_layout = with_window_fields(spec.source_layout, start, end);
		spec.target_layout = with_window_fields(spec.target_layout, start, end);
	};

	SECTION("unaligned start") {
		window(3, 288);
		const auto set = apply_chunking(spec, chunking_strategy(64));
		CHECK(implements(set, spec));
		CHECK(chunk_lengths(set) == std::vector<int64_t>{61, 64, 64, 64, 32}); // [3,64), [64,128), ..., [256,288)
	}

	SECTION("unaligned start and end") {
		window(3, 285);
		const auto set = apply_chunking(spec, chunking_strategy(64));
		CHECK(implements(set, spec));
		CHECK(chunk_lengths(set) == std::vector<int64_t>{61, 64, 64, 64, 29});
	}

	SECTION("every interior boundary is aligned") {
		window(3, 285);
		const auto chunks = chunks_by_start(apply_chunking(spec, chunking_strategy(64)));
		REQUIRE(chunks.size() > 1);
		for(size_t i = 1; i < chunks.size(); ++i) {
			CHECK(chunks[i].source_layout.start % ref::alignment == 0);
		}
	}
}

TEST_CASE("a partial window is chunked within the window", "[chunking][window]") {
	auto spec = ref::spec();
	spec.source_layout = with_window_fields(spec.source_layout, 24, 264);
	spec.target_layout = with_window_fields(spec.target_layout, 24, 264);

	const auto set = apply_chunking(spec, chunking_strategy(64));

	CHECK(implements(set, spec));
	const auto chunks = chunks_by_start(set);
	REQUIRE_FALSE(chunks.empty());
	CHECK(chunks.front().source_layout.start == 24);
	CHECK(chunks.back().source_layout.end == 264);
}

TEST_CASE("a reshaping spec is chunked on both sides", "[chunking]") {
	// source row extent 24, target row extent 12, strides 24 / 24 and 12 / 24: the alignment is 4
	const auto spec = reshaping_spec();
	const auto set = apply_chunking(spec, chunking_strategy(8));

	CHECK(implements(set, spec));
	CHECK(chunk_lengths(set) == std::vector<int64_t>{8, 8, 8});
}

// ---------------------------------------------------------------------------------------------------------------------
// copy_alignment: every term matters, each case below goes wrong without its own

TEST_CASE("the alignment of the documented specs", "[chunking][alignment]") {
	CHECK(copy_alignment(ref::spec()) == ref::alignment); // row extent 24, d0_stride 80, d0_start_offset 16
	CHECK(copy_alignment(reshaping_spec()) == 4);         // the 12-byte rows of the target
}

TEST_CASE("the alignment accounts for every term on both sides", "[chunking][alignment]") {
	// 64-byte rows starting at byte 0 of 128-byte rows, which on their own allow 64
	const auto clean = layout_from_fields(0x10000, 128, 4, 0, 0, 0, 64, 4, 1, 0, 256);
	const auto on_both = [](const data_layout& l) { return spec_from_fields(device_id::d0, l, device_id::d1, l); };
	REQUIRE(copy_alignment(on_both(clean)) == 64);

	SECTION("row extent") { CHECK(copy_alignment(on_both(layout_from_fields(0x10000, 64, 4, 0, 0, 0, 48, 4, 1, 0, 192))) == 16); }
	SECTION("d0_stride") { CHECK(copy_alignment(on_both(layout_from_fields(0x10000, 96, 4, 0, 0, 0, 64, 4, 1, 0, 256))) == 32); }
	SECTION("d0_start_offset, on either side alone") {
		// every row starts 4 bytes past a 64-byte boundary
		const auto at_4 = layout_from_fields(0x20000, 128, 4, 4, 0, 0, 68, 4, 1, 0, 256);
		CHECK(copy_alignment(spec_from_fields(device_id::d0, at_4, device_id::d1, clean)) == 4);
		CHECK(copy_alignment(spec_from_fields(device_id::d0, clean, device_id::d1, at_4)) == 4);
	}
	SECTION("the shift between the windows") {
		CHECK(copy_alignment(spec_from_fields(device_id::d0, with_window_fields(clean, 0, 248), device_id::d1, with_window_fields(clean, 8, 256))) == 8);
	}
	SECTION("but not an equal shift of both windows") {
		// only the first and last chunk are ragged then, interior boundaries stay aligned on both sides
		CHECK(copy_alignment(spec_from_fields(device_id::d0, with_window_fields(clean, 3, 256), device_id::d1, with_window_fields(clean, 3, 256))) == 64);
	}
	SECTION("odd rows fall back to single bytes") { CHECK(copy_alignment(on_both(layout_from_fields(0x10000, 9, 4, 0, 0, 0, 9, 4, 1, 0, 36))) == 1); }
}

TEST_CASE("chunking rejects an invalid spec", "[chunking][error]") {
	auto invalid = ref::spec();
	invalid.target_layout.end -= 8; // window lengths differ
	REQUIRE_FALSE(is_valid(invalid));

	CHECK_THROWS_AS(apply_chunking(invalid, chunking_strategy(64)), copylib::error);
	CHECK_NOTHROW(apply_chunking(ref::spec(), chunking_strategy(64)));
}
