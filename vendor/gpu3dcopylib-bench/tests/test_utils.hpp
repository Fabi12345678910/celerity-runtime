#pragma once

#include <copylib/core.hpp>
#include <copylib/support.hpp>

#include <catch2/catch_tostring.hpp>

#include <algorithm>
#include <compare>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// Catch2 prints the operands of a failed comparison through these, so failures show layouts and specs through the
// library's formatters.
namespace Catch {
template <>
struct StringMaker<copylib::data_layout> {
	static std::string convert(const copylib::data_layout& v) { return copylib::utils::format("{}", v); }
};
template <>
struct StringMaker<copylib::copy_spec> {
	static std::string convert(const copylib::copy_spec& v) { return copylib::utils::format("{}", v); }
};
template <>
struct StringMaker<copylib::copy_plan> {
	static std::string convert(const copylib::copy_plan& v) { return copylib::utils::format("{}", v); }
};
template <>
struct StringMaker<copylib::parallel_copy_set> {
	static std::string convert(const copylib::parallel_copy_set& v) { return copylib::utils::format("{}", v); }
};
template <>
struct StringMaker<copylib::copy_strategy> {
	static std::string convert(const copylib::copy_strategy& v) { return copylib::utils::format("{}", v); }
};
} // namespace Catch

namespace copylib_testing {

// ---------------------------------------------------------------------------------------------------------------------
// Construction helpers that bypass the constructors, so that a failure in the layers above them points at one layer
// instead of cascading from the constructors.

[[nodiscard]] inline copylib::data_layout layout_from_fields(intptr_t base, int64_t d0_stride, int64_t d1_stride, int64_t d0_start_offset,
    int64_t d1_start_offset, int64_t d2_start_offset, int64_t d0_end_offset, int64_t d1_end_offset, int64_t d2_end_offset, int64_t start, int64_t end) {
	copylib::data_layout layout;
	layout.base = base;
	layout.d0_stride = d0_stride;
	layout.d1_stride = d1_stride;
	layout.d0_start_offset = d0_start_offset;
	layout.d1_start_offset = d1_start_offset;
	layout.d2_start_offset = d2_start_offset;
	layout.d0_end_offset = d0_end_offset;
	layout.d1_end_offset = d1_end_offset;
	layout.d2_end_offset = d2_end_offset;
	layout.start = start;
	layout.end = end;
	return layout;
}

// the same for staging ids
[[nodiscard]] inline copylib::staging_id staging_id_from_fields(bool on_host, copylib::device_id did, uint32_t index) {
	copylib::staging_id id;
	id.on_host = on_host;
	id.did = did;
	id.index = index;
	return id;
}

// a contiguous 1D layout of `length` bytes at the start of an unplaced staging buffer
[[nodiscard]] inline copylib::data_layout staging_layout_from_fields(copylib::staging_id staging, int64_t length) {
	auto layout = layout_from_fields(0, length, 1, 0, 0, 0, length, 1, 1, 0, length);
	layout.staging = staging;
	return layout;
}

// the same layout with only its window changed
[[nodiscard]] inline copylib::data_layout with_window_fields(copylib::data_layout layout, int64_t start, int64_t end) {
	layout.start = start;
	layout.end = end;
	return layout;
}

// Likewise for copy specs. copy_spec has no default constructor, so the fields are assigned again after constructing.
[[nodiscard]] inline copylib::copy_spec spec_from_fields(
    copylib::device_id source_device, const copylib::data_layout& source, copylib::device_id target_device, const copylib::data_layout& target) {
	copylib::copy_spec spec{source_device, source, target_device, target};
	spec.source_device = source_device;
	spec.source_layout = source;
	spec.target_device = target_device;
	spec.target_layout = target;
	return spec;
}

// and for strategies
[[nodiscard]] inline copylib::copy_strategy strategy_from_fields(
    copylib::copy_type type, copylib::copy_properties properties, copylib::d2d_implementation d2d, int64_t chunk_size) {
	copylib::copy_strategy strategy;
	strategy.type = type;
	strategy.properties = properties;
	strategy.d2d = d2d;
	strategy.chunk_size = chunk_size;
	return strategy;
}

[[nodiscard]] inline copylib::copy_strategy chunking_strategy(int64_t chunk_size) {
	return strategy_from_fields(copylib::copy_type::direct, copylib::copy_properties::none, copylib::d2d_implementation::direct, chunk_size);
}

// ---------------------------------------------------------------------------------------------------------------------
// Field-wise comparisons, independent of the library's operator==.

// The base/staging union is the first member of data_layout, so copying its object representation is well-defined and
// avoids reading an inactive union member.
[[nodiscard]] inline intptr_t space_of(const copylib::data_layout& layout) {
	intptr_t bits = 0;
	std::memcpy(&bits, &layout, sizeof bits);
	return bits;
}

// Unplaced staging buffers are recognised by their first byte, exactly as design.md describes it.
[[nodiscard]] inline bool is_staging_space(const copylib::data_layout& layout) {
	uint8_t first_byte = 0;
	std::memcpy(&first_byte, &layout, 1);
	return first_byte == copylib::staging_id::staging_id_flag;
}

[[nodiscard]] inline bool same_box(const copylib::data_layout& a, const copylib::data_layout& b) {
	return space_of(a) == space_of(b) && a.d0_stride == b.d0_stride && a.d1_stride == b.d1_stride && a.d0_start_offset == b.d0_start_offset
	       && a.d1_start_offset == b.d1_start_offset && a.d2_start_offset == b.d2_start_offset && a.d0_end_offset == b.d0_end_offset
	       && a.d1_end_offset == b.d1_end_offset && a.d2_end_offset == b.d2_end_offset;
}

[[nodiscard]] inline bool same_fields(const copylib::data_layout& a, const copylib::data_layout& b) {
	return same_box(a, b) && a.start == b.start && a.end == b.end;
}

[[nodiscard]] inline int64_t row_extent_of(const copylib::data_layout& l) { return l.d0_end_offset - l.d0_start_offset; }

[[nodiscard]] inline int64_t rows_in_box(const copylib::data_layout& l) {
	return (l.d1_end_offset - l.d1_start_offset) * (l.d2_end_offset - l.d2_start_offset);
}

// ---------------------------------------------------------------------------------------------------------------------
// Reference interpretation of layouts. Everything the library computes about a layout is checked against this, so it
// is written as the most direct loop over the raw fields, sharing no code with the library.

// Allocation offsets (relative to the base) of every byte in the layout's window, in packed order.
// Returns an empty vector for layouts that do not describe a non-empty box and window.
[[nodiscard]] inline std::vector<int64_t> reference_offsets(const copylib::data_layout& l) {
	const int64_t row_extent = l.d0_end_offset - l.d0_start_offset;
	const int64_t rows = l.d1_end_offset - l.d1_start_offset;
	const int64_t planes = l.d2_end_offset - l.d2_start_offset;
	if(l.d0_stride <= 0 || l.d1_stride <= 0 || row_extent <= 0 || rows <= 0 || planes <= 0) return {};
	const int64_t plane_bytes = l.d1_stride * l.d0_stride; // in the allocation
	const int64_t box_plane_bytes = row_extent * rows;     // in the box, gaps excluded
	if(l.start < 0 || l.end > box_plane_bytes * planes || l.start >= l.end) return {};

	std::vector<int64_t> offsets;
	offsets.reserve(static_cast<size_t>(l.end - l.start));
	for(int64_t p = l.start; p < l.end; ++p) {
		const int64_t plane = p / box_plane_bytes;
		const int64_t row = (p % box_plane_bytes) / row_extent;
		const int64_t col = p % row_extent;
		offsets.push_back((plane + l.d2_start_offset) * plane_bytes + (row + l.d1_start_offset) * l.d0_stride + col + l.d0_start_offset);
	}
	return offsets;
}

// Index of the first difference between two offset lists, or -1 if they are equal. Comparing the vectors directly
// would make Catch2 print every element on failure, which for real layouts is thousands of numbers.
[[nodiscard]] inline int64_t first_difference(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
	const size_t common = std::min(a.size(), b.size());
	for(size_t i = 0; i < common; ++i) {
		if(a[i] != b[i]) return static_cast<int64_t>(i);
	}
	return a.size() == b.size() ? -1 : static_cast<int64_t>(common);
}

struct reference_run {
	int64_t offset;
	int64_t length;
};

// The maximal contiguous runs of the window: adjacent rows merge into one run.
[[nodiscard]] inline std::vector<reference_run> reference_runs(const copylib::data_layout& l) {
	std::vector<reference_run> runs;
	for(const auto offset : reference_offsets(l)) {
		if(!runs.empty() && runs.back().offset + runs.back().length == offset) {
			++runs.back().length;
		} else {
			runs.push_back({offset, 1});
		}
	}
	return runs;
}

// ---------------------------------------------------------------------------------------------------------------------
// Symbolic execution of copy plans and sets. Instead of moving data, it tracks for every byte written where its value
// originally came from. That makes it an oracle for all planning functions: a plan implements a spec exactly when every
// byte of the spec's target window ends up holding the corresponding byte of its source window, and nothing else
// outside of staging buffers was written.

struct address {
	copylib::device_id device;
	intptr_t space; // allocation base, or the bit pattern of an unplaced staging id
	int64_t offset;

	friend auto operator<=>(const address&, const address&) = default;
};

using byte_mapping = std::map<address, address>; // written address -> address its value originates from

struct simulation {
	byte_mapping final_state; // every non-staging address that was written, and where its value came from
	bool consistent = true;   // false if a step had malformed layouts or windows of different lengths
	bool conflicting = false; // true if two plans of a set wrote the same address, staging buffers included
	int64_t max_writes = 0;   // the most writes any single non-staging address received
};

[[nodiscard]] inline simulation simulate(const copylib::parallel_copy_set& set) {
	simulation sim;
	byte_mapping state;
	std::map<address, size_t> last_writer;
	std::map<address, int64_t> writes;
	std::map<address, bool> staging_address;

	for(size_t plan_idx = 0; plan_idx < set.size(); ++plan_idx) {
		for(const auto& spec : set[plan_idx]) {
			const auto source = reference_offsets(spec.source_layout);
			const auto target = reference_offsets(spec.target_layout);
			if(source.empty() || source.size() != target.size()) {
				sim.consistent = false;
				continue;
			}
			const auto source_space = space_of(spec.source_layout);
			const auto target_space = space_of(spec.target_layout);
			const bool target_is_staging = is_staging_space(spec.target_layout);

			for(size_t i = 0; i < source.size(); ++i) {
				const address from{spec.source_device, source_space, source[i]};
				const address to{spec.target_device, target_space, target[i]};
				// a byte nobody wrote still holds its original value; for a staging buffer that is garbage, which then
				// never matches the expected source address
				const auto known = state.find(from);
				state[to] = known != state.end() ? known->second : from;

				const auto writer = last_writer.find(to);
				if(writer != last_writer.end() && writer->second != plan_idx) { sim.conflicting = true; }
				last_writer[to] = plan_idx;
				++writes[to];
				staging_address[to] = target_is_staging;
			}
		}
	}

	for(const auto& [addr, origin] : state) {
		if(staging_address[addr]) continue;
		sim.final_state.emplace(addr, origin);
		sim.max_writes = std::max(sim.max_writes, writes[addr]);
	}
	return sim;
}

[[nodiscard]] inline simulation simulate(const copylib::copy_plan& plan) { return simulate(copylib::parallel_copy_set{plan}); }

// What executing a spec should leave behind.
[[nodiscard]] inline byte_mapping expected_mapping(const copylib::copy_spec& spec) {
	byte_mapping mapping;
	const auto source = reference_offsets(spec.source_layout);
	const auto target = reference_offsets(spec.target_layout);
	if(source.size() != target.size()) return mapping;
	const auto source_space = space_of(spec.source_layout);
	const auto target_space = space_of(spec.target_layout);
	for(size_t i = 0; i < source.size(); ++i) {
		mapping[{spec.target_device, target_space, target[i]}] = {spec.source_device, source_space, source[i]};
	}
	return mapping;
}

// The ground truth for the planning layers: correct result, every target byte written exactly once (overlapping chunks
// do not count even though they produce the right bytes), and no two plans of a set touching the same memory.
[[nodiscard]] inline bool implements(const simulation& sim, const copylib::copy_spec& spec) {
	return sim.consistent && !sim.conflicting && sim.max_writes == 1 && !sim.final_state.empty() && sim.final_state == expected_mapping(spec);
}

template <typename PlanOrSet>
[[nodiscard]] bool implements(const PlanOrSet& plan_or_set, const copylib::copy_spec& spec) {
	return implements(simulate(plan_or_set), spec);
}

// ---------------------------------------------------------------------------------------------------------------------
// Helpers for the planning layers above chunking.

struct staging_request {
	copylib::device_id did;
	bool on_host;
	int64_t size;
	copylib::staging_id id;
};

// A staging provider that hands out a fresh id per request and logs what was asked for. The id carries the requested
// device and host flag, so a layout's id tells where its buffer lives. The log must outlive the provider.
[[nodiscard]] inline copylib::staging_buffer_provider recording_provider(std::vector<staging_request>& log) {
	return [&log](copylib::device_id did, bool on_host, int64_t size) {
		const auto id = staging_id_from_fields(on_host, did, static_cast<uint32_t>(log.size()));
		log.push_back({did, on_host, size, id});
		return id;
	};
}

// the staging id of an unplaced staging layout, read without touching an inactive union member
[[nodiscard]] inline copylib::staging_id staging_of(const copylib::data_layout& layout) {
	copylib::staging_id id;
	std::memcpy(static_cast<void*>(&id), &layout, sizeof id); // staging_id is trivially copyable, just not trivial
	return id;
}

// the exact encoding of a staging buffer holding `length` bytes: 1D, sized to fit, window [0, length)
[[nodiscard]] inline bool is_staging_buffer(const copylib::data_layout& layout, int64_t length) {
	return is_staging_space(layout) && same_fields(layout, staging_layout_from_fields(staging_of(layout), length));
}

// whether a window is one contiguous run, judged by the oracle rather than by the library
[[nodiscard]] inline bool is_one_run(const copylib::data_layout& layout) { return reference_runs(layout).size() == 1; }

[[nodiscard]] inline bool crosses_devices(const copylib::copy_spec& spec) { return spec.source_device != spec.target_device; }

[[nodiscard]] inline bool is_device_to_device(const copylib::copy_spec& spec) {
	return crosses_devices(spec) && spec.source_device != copylib::device_id::host && spec.target_device != copylib::device_id::host;
}

[[nodiscard]] inline bool same_spec(const copylib::copy_spec& a, const copylib::copy_spec& b) {
	return a.source_device == b.source_device && a.target_device == b.target_device && a.properties == b.properties
	       && same_fields(a.source_layout, b.source_layout) && same_fields(a.target_layout, b.target_layout);
}

[[nodiscard]] inline bool same_plan(const copylib::copy_plan& a, const copylib::copy_plan& b) {
	return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), same_spec);
}

[[nodiscard]] inline bool same_set(const copylib::parallel_copy_set& a, const copylib::parallel_copy_set& b) {
	return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), same_plan);
}

// each step reads exactly what the previous one wrote, on the same device
[[nodiscard]] inline bool steps_connect(const copylib::copy_plan& plan) {
	for(size_t i = 1; i < plan.size(); ++i) {
		if(plan[i - 1].target_device != plan[i].source_device || !same_fields(plan[i - 1].target_layout, plan[i].source_layout)) return false;
	}
	return true;
}

// Every staging buffer is used with one size, and each step reaches it from where it lives: a host buffer from the
// host, a device buffer from its own device.
[[nodiscard]] inline bool staging_is_consistent(const copylib::parallel_copy_set& set) {
	std::map<intptr_t, int64_t> size_of;
	const auto consistent = [&](const copylib::data_layout& layout, copylib::device_id step_device) {
		if(!is_staging_space(layout)) return true;
		const auto id = staging_of(layout);
		if(step_device != (id.on_host ? copylib::device_id::host : id.did)) return false;
		const auto [it, inserted] = size_of.emplace(space_of(layout), layout.total_bytes());
		return inserted || it->second == layout.total_bytes();
	};
	for(const auto& plan : set) {
		for(const auto& spec : plan) {
			if(!consistent(spec.source_layout, spec.source_device) || !consistent(spec.target_layout, spec.target_device)) return false;
		}
	}
	return true;
}

[[nodiscard]] inline bool staging_is_consistent(const copylib::copy_plan& plan) { return staging_is_consistent(copylib::parallel_copy_set{plan}); }

[[nodiscard]] inline bool all_steps_have(const copylib::parallel_copy_set& set, copylib::copy_properties properties) {
	for(const auto& plan : set) {
		for(const auto& spec : plan) {
			if(spec.properties != properties) return false;
		}
	}
	return true;
}

[[nodiscard]] inline bool all_steps_have(const copylib::copy_plan& plan, copylib::copy_properties properties) {
	return all_steps_have(copylib::parallel_copy_set{plan}, properties);
}

// ---------------------------------------------------------------------------------------------------------------------
// The reference layout from docs/design.md: the box [2,5) x [3,7) x [4,10) of an `int[12,16,20]` allocation.
// d0 is the innermost (contiguous) dimension, d2 the outermost, and all layout values are in bytes.
namespace reference_box {

	constexpr int64_t elem_size = sizeof(int32_t); // 4

	constexpr int64_t d0_extent = 20; // elements per row
	constexpr int64_t d1_extent = 16; // rows per plane

	constexpr int64_t d0_stride = d0_extent * elem_size;   // 80 bytes, one full row
	constexpr int64_t d1_stride = d1_extent;               // 16 rows, one full plane
	constexpr int64_t plane_bytes = d1_stride * d0_stride; // 1280

	constexpr int64_t d0_start_offset = 4 * elem_size; // 16 bytes
	constexpr int64_t d0_end_offset = 10 * elem_size;  // 40 bytes
	constexpr int64_t d1_start_offset = 3;             // rows
	constexpr int64_t d1_end_offset = 7;
	constexpr int64_t d2_start_offset = 2; // planes
	constexpr int64_t d2_end_offset = 5;

	constexpr int64_t row_extent = d0_end_offset - d0_start_offset; // 24 bytes
	constexpr int64_t rows = d1_end_offset - d1_start_offset;       // 4
	constexpr int64_t planes = d2_end_offset - d2_start_offset;     // 3
	constexpr int64_t total_bytes = row_extent * rows * planes;     // 288

	// offset of the first byte of the box relative to the allocation base
	constexpr int64_t first_byte = d2_start_offset * plane_bytes + d1_start_offset * d0_stride + d0_start_offset; // 2816

	// largest power of two up to 64 dividing the row extent, d0_stride and d0_start_offset (24, 80, 16)
	constexpr int64_t alignment = 8;

	// a base address that is safely distinguishable from a staging_id (which is detected via its lowest byte)
	constexpr intptr_t base = 0x10000;

	// built through the constructor, so that constructor tests exercise it
	[[nodiscard]] inline copylib::data_layout make() {
		return copylib::data_layout{base, d0_stride, d1_stride, d0_start_offset, d1_start_offset, d2_start_offset, d0_end_offset, d1_end_offset, d2_end_offset};
	}

	// built by assigning fields, for tests that are not about the constructors
	[[nodiscard]] inline copylib::data_layout fields(intptr_t at = base) {
		return layout_from_fields(
		    at, d0_stride, d1_stride, d0_start_offset, d1_start_offset, d2_start_offset, d0_end_offset, d1_end_offset, d2_end_offset, 0, total_bytes);
	}

	// the reference box on d0 copied to the same box of another allocation on d1
	[[nodiscard]] inline copylib::copy_spec spec() { return spec_from_fields(copylib::device_id::d0, fields(base), copylib::device_id::d1, fields(0x80000)); }

} // namespace reference_box

// Further layouts within the same int[12,16,20] allocation.
namespace shapes {

	namespace ref = reference_box;

	// the whole allocation: every row of every plane, no gaps
	[[nodiscard]] inline copylib::data_layout whole_allocation(intptr_t at = ref::base) {
		constexpr int64_t planes = 12;
		return layout_from_fields(at, ref::d0_stride, ref::d1_stride, 0, 0, 0, ref::d0_stride, ref::d1_stride, planes, 0, planes * ref::plane_bytes);
	}

	// a single partial row of the first plane
	[[nodiscard]] inline copylib::data_layout single_row(intptr_t at = ref::base) {
		return layout_from_fields(at, ref::d0_stride, ref::d1_stride, ref::d0_start_offset, 0, 0, ref::d0_end_offset, 1, 1, 0, ref::row_extent);
	}

	// four full rows of the first plane, which form one contiguous run
	[[nodiscard]] inline copylib::data_layout full_rows_of_one_plane(intptr_t at = ref::base) {
		return layout_from_fields(
		    at, ref::d0_stride, ref::d1_stride, 0, ref::d1_start_offset, 0, ref::d0_stride, ref::d1_end_offset, 1, 0, ref::rows * ref::d0_stride);
	}

	// four full rows in each of three planes: one contiguous run per plane
	[[nodiscard]] inline copylib::data_layout full_rows_of_three_planes(intptr_t at = ref::base) {
		return layout_from_fields(at, ref::d0_stride, ref::d1_stride, 0, ref::d1_start_offset, ref::d2_start_offset, ref::d0_stride, ref::d1_end_offset,
		    ref::d2_end_offset, 0, ref::rows * ref::d0_stride * ref::planes);
	}

	// partial rows spanning all 16 rows of two consecutive planes: since a plane is exactly 16 rows, the rows continue
	// across the plane boundary with the same spacing, even though no two of them are adjacent
	[[nodiscard]] inline copylib::data_layout partial_rows_of_two_full_planes(intptr_t at = ref::base) {
		return layout_from_fields(
		    at, ref::d0_stride, ref::d1_stride, ref::d0_start_offset, 0, 0, ref::d0_end_offset, ref::d1_stride, 2, 0, ref::row_extent * ref::d1_extent * 2);
	}

	// 288 bytes as six rows of 48 inside rows of 64: the length of the reference box in a different strided shape
	[[nodiscard]] inline copylib::data_layout six_rows_of_48(intptr_t at) { return layout_from_fields(at, 64, 8, 0, 1, 0, 48, 7, 1, 0, 288); }

	// int[1,1,6] and int[1,2,3], the reshaping pair from design.md
	[[nodiscard]] inline copylib::data_layout one_row_of_six(intptr_t at) {
		constexpr int64_t e = ref::elem_size;
		return layout_from_fields(at, 6 * e, 1, 0, 0, 0, 6 * e, 1, 1, 0, 6 * e);
	}
	[[nodiscard]] inline copylib::data_layout two_rows_of_three(intptr_t at) {
		constexpr int64_t e = ref::elem_size;
		return layout_from_fields(at, 3 * e, 2, 0, 0, 0, 3 * e, 2, 1, 0, 6 * e);
	}

} // namespace shapes

// Builders for the normal forms defined in docs/design.md ("Normal form and equality"), so that tests state expected
// encodings in the same terms as the design table. Windows cover the whole box.
namespace normal_form {

	// 1D: a single contiguous run of `length` bytes at `offset`; the row stride is the end of the run
	[[nodiscard]] inline copylib::data_layout one_run(intptr_t at, int64_t offset, int64_t length) {
		return layout_from_fields(at, offset + length, 1, offset, 0, 0, offset + length, 1, 1, 0, length);
	}

	// 2D: `count` runs of `length` bytes, the first at `offset`, spaced `spacing` bytes apart
	[[nodiscard]] inline copylib::data_layout uniform_runs(intptr_t at, int64_t offset, int64_t length, int64_t count, int64_t spacing) {
		const int64_t in_row = offset % spacing;
		const int64_t first_row = offset / spacing;
		const int64_t rows = first_row + count;
		return layout_from_fields(at, spacing, rows, in_row, first_row, 0, in_row + length, rows, 1, 0, count * length);
	}

} // namespace normal_form

} // namespace copylib_testing
