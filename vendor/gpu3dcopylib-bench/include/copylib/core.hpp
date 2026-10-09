#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace copylib {

enum class device_id : int16_t {
	host = -1,
	d0 = 0,
	d1 = 1,
	d2 = 2,
	d3 = 3,
	d4 = 4,
	d5 = 5,
	d6 = 6,
	d7 = 7,
	count = 8,
};

#pragma pack(push, 0)
struct staging_id {
	constexpr static uint8_t staging_id_flag = 0b00000001;
	uint8_t is_staging_id = staging_id_flag;
	uint8_t on_host = false;
	device_id did = device_id::d0;
	uint32_t index = 0;

	staging_id() = default;
	staging_id(bool on_host, device_id did, uint32_t index) : on_host(on_host), did(did), index(index) {}

	constexpr bool operator==(const staging_id& other) const = default;
	constexpr bool operator!=(const staging_id& other) const = default;
};
#pragma pack(pop)
static_assert(sizeof(staging_id) == sizeof(intptr_t));
static_assert(offsetof(staging_id, is_staging_id) == 0);

// 3D data layout used as the source or destination of a copy operation
// d0 is the innermost dimension and always contiguous, d2 the outermost.
// d0 counts bytes, d1 rows and d2 planes, so a plane spans d1_stride * d0_stride bytes.
struct data_layout {
	union {
		intptr_t base = 0;
		staging_id staging;
	};

	int64_t d0_stride = 0; // bytes per row of the allocation
	int64_t d1_stride = 0; // rows per plane of the allocation

	int64_t d0_start_offset = 0; // bytes into the row
	int64_t d1_start_offset = 0; // rows into the plane
	int64_t d2_start_offset = 0; // planes into the allocation

	int64_t d0_end_offset = 0;
	int64_t d1_end_offset = 0;
	int64_t d2_end_offset = 0;

	// half-open window of byte offsets into the packed box, i.e. with the gaps excluded, with 0 <= start < end <= total_bytes().
	// The constructors set it to cover the whole box; chunking keeps the box and narrows only the window.
	int64_t start = 0;
	int64_t end = 0;

	data_layout() = default;

	// a contiguous 1D layout of `length` bytes, starting `offset` bytes into the allocation
	data_layout(intptr_t base, int64_t offset, int64_t length)
	    : base(base), d0_stride(length + offset), d1_stride(1), d0_start_offset(offset), d1_start_offset(0), d2_start_offset(0), d0_end_offset(length + offset),
	      d1_end_offset(1), d2_end_offset(1), start(0), end(length) {}
	data_layout(intptr_t base, int64_t d0_stride, int64_t d1_stride, int64_t d0_start_offset, int64_t d1_start_offset, int64_t d2_start_offset,
	    int64_t d0_end_offset, int64_t d1_end_offset, int64_t d2_end_offset)
	    : base(base), d0_stride(d0_stride), d1_stride(d1_stride), d0_start_offset(d0_start_offset), d1_start_offset(d1_start_offset),
	      d2_start_offset(d2_start_offset), d0_end_offset(d0_end_offset), d1_end_offset(d1_end_offset), d2_end_offset(d2_end_offset), start(0),
	      end((d2_end_offset - d2_start_offset) * (d1_end_offset - d1_start_offset) * (d0_end_offset - d0_start_offset)) {}
	// the same box and window as `layout`, but in the allocation at `base`
	data_layout(intptr_t base, const data_layout& layout)
	    : base(base), d0_stride(layout.d0_stride), d1_stride(layout.d1_stride), d0_start_offset(layout.d0_start_offset),
	      d1_start_offset(layout.d1_start_offset), d2_start_offset(layout.d2_start_offset), d0_end_offset(layout.d0_end_offset),
	      d1_end_offset(layout.d1_end_offset), d2_end_offset(layout.d2_end_offset), start(layout.start), end(layout.end) {}

	data_layout(staging_id staging, int64_t offset, int64_t length)
	    : staging(staging), d0_stride(length + offset), d1_stride(1), d0_start_offset(offset), d1_start_offset(0), d2_start_offset(0),
	      d0_end_offset(length + offset), d1_end_offset(1), d2_end_offset(1), start(0), end(length) {}
	data_layout(staging_id staging, int64_t d0_stride, int64_t d1_stride, int64_t d0_start_offset, int64_t d1_start_offset, int64_t d2_start_offset,
	    int64_t d0_end_offset, int64_t d1_end_offset, int64_t d2_end_offset)
	    : staging(staging), d0_stride(d0_stride), d1_stride(d1_stride), d0_start_offset(d0_start_offset), d1_start_offset(d1_start_offset),
	      d2_start_offset(d2_start_offset), d0_end_offset(d0_end_offset), d1_end_offset(d1_end_offset), d2_end_offset(d2_end_offset), start(0),
	      end((d2_end_offset - d2_start_offset) * (d1_end_offset - d1_start_offset) * (d0_end_offset - d0_start_offset)) {}
	// the same box and window as `layout`, but in the unplaced staging buffer `staging`
	data_layout(staging_id staging, const data_layout& layout)
	    : staging(staging), d0_stride(layout.d0_stride), d1_stride(layout.d1_stride), d0_start_offset(layout.d0_start_offset),
	      d1_start_offset(layout.d1_start_offset), d2_start_offset(layout.d2_start_offset), d0_end_offset(layout.d0_end_offset),
	      d1_end_offset(layout.d1_end_offset), d2_end_offset(layout.d2_end_offset), start(layout.start), end(layout.end) {}

	// the same layout with the window narrowed to [start, end)
	[[nodiscard]] data_layout with_window(int64_t start, int64_t end) const {
		data_layout layout(this->base, *this);
		layout.start = start;
		layout.end = end;
		return layout;
	}

	// the number of bytes this layout actually copies, i.e. the length of its window
	[[nodiscard]] constexpr int64_t window_length() const { return end - start; }

	// bytes covered by the box, excluding the gaps
	[[nodiscard]] constexpr int64_t total_bytes() const {
		return (d2_end_offset - d2_start_offset) * (d1_end_offset - d1_start_offset) * (d0_end_offset - d0_start_offset);
	}

	// byte offset in the allocation of the byte `packed_offset` bytes into the box, with the gaps excluded.
	// This is the closed form the copy kernels need: byte i of a copy sits at offset_at(start + i).
	[[nodiscard]] constexpr int64_t offset_at(int64_t packed_offset) const {
		int64_t d0_size = d0_end_offset - d0_start_offset;
		int64_t d1_size = d0_size * (d1_end_offset - d1_start_offset);
		int64_t i_d0 = packed_offset % d0_size;
		int64_t i_d1 = (packed_offset % d1_size) / d0_size;
		int64_t i_d2 = packed_offset / d1_size;
		// byte packed_offset sits at box[i_d2][i_d1][i_d0]
		return (i_d2 + d2_start_offset) * d1_stride * d0_stride + (i_d1 + d1_start_offset) * d0_stride + (i_d0 + d0_start_offset);
	}

	// rows are adjacent, so they can be collapsed
	[[nodiscard]] constexpr bool d1_contiguous() const { return d0_start_offset == 0 && d0_end_offset == d0_stride; }
	// planes are adjacent, so they can be collapsed
	[[nodiscard]] constexpr bool d2_contiguous() const { return d1_contiguous() && (d1_start_offset == 0 && d1_end_offset == d1_stride); }

	// shape predicates, all defined over the window rather than over the whole box
	[[nodiscard]] constexpr bool is_window_contiguous() const {
		if(d2_contiguous()) return true; // the whole box is one run
		const int64_t d0_extent = d0_end_offset - d0_start_offset;
		const int64_t last = end - 1;
		if(d1_contiguous()) { // each plane is one run
			const int64_t d1_extent = d0_extent * (d1_end_offset - d1_start_offset);
			return start / d1_extent == last / d1_extent;
		}
		return start / d0_extent == last / d0_extent; // inside a single row
	}

	[[nodiscard]] constexpr bool is_unplaced_staging() const { return staging.is_staging_id == staging_id::staging_id_flag; }
	[[nodiscard]] std::byte* base_ptr() const { return reinterpret_cast<std::byte*>(base); }

	// Compares the encoding field by field, window included. Two layouts describing the same bytes with different strides
	// compare unequal; normalize both first to compare the bytes they describe.
	constexpr bool operator==(const data_layout& other) const {
		return other.d0_stride == d0_stride && other.d1_stride == d1_stride && other.d0_end_offset == d0_end_offset && other.d1_end_offset == d1_end_offset
		       && other.d2_end_offset == d2_end_offset && other.d0_start_offset == d0_start_offset && other.d1_start_offset == d1_start_offset
		       && other.d2_start_offset == d2_start_offset && other.start == start && other.end == end && other.base == base;
	}

	constexpr bool operator!=(const data_layout& other) const { return !(*this == other); }
};

enum class copy_properties {
	none = 0x0000,
	use_kernel = 0x0001, // whether to use a kernel to perform the copy
};

inline copy_properties operator|(copy_properties a, copy_properties b) { return static_cast<copy_properties>(static_cast<int>(a) | static_cast<int>(b)); }
inline bool operator&(copy_properties a, copy_properties b) { return static_cast<int>(a) & static_cast<int>(b); }

// a copy specification describes a single copy operation from a source data layout and device to a destination data layout and device
struct copy_spec {
	device_id source_device;
	data_layout source_layout;
	device_id target_device;
	data_layout target_layout;

	copy_properties properties = copy_properties::none;

	constexpr copy_spec(device_id src_dev, const data_layout& src_layout, device_id tgt_dev, const data_layout& tgt_layout)
	    : source_device(src_dev), source_layout(src_layout), target_device(tgt_dev), target_layout(tgt_layout) {}
	constexpr copy_spec(device_id src_dev, const data_layout& src_layout, device_id tgt_dev, const data_layout& tgt_layout, copy_properties p)
	    : source_device(src_dev), source_layout(src_layout), target_device(tgt_dev), target_layout(tgt_layout), properties(p) {}

	[[nodiscard]] constexpr copy_spec with_properties(copy_properties p) const { return {source_device, source_layout, target_device, target_layout, p}; }

	constexpr bool operator==(const copy_spec&) const = default;
	constexpr bool operator!=(const copy_spec&) const = default;
};

// a copy plan is a list of one or more copy specifications which need to be enacted subsequently to implement one semantic copy operation
using copy_plan = std::vector<copy_spec>;

// a parallel copy set is a set of independent copy plans which can be enacted concurrently
using parallel_copy_set = std::vector<copy_plan>;

// defines the strategy type used to copy data between memories
enum class copy_type {
	direct, // copy directly from source to destination using copy operations
	staged, // stage/unstage to a linearized buffer to perform the copy
};

// defines how to deal with device to device copy operations
enum class d2d_implementation {
	direct,                 // directly copy from device to device
	host_staging_at_source, // stage in host memory at the source device
	host_staging_at_target, // stage in host memory at the target device
	host_staging_at_both,   // stage in host memory at both devices, with extra copy operation
};

// defines the strategy used to copy data between memories
struct copy_strategy {
	copy_type type = copy_type::direct;
	copy_properties properties = copy_properties::none;
	d2d_implementation d2d = d2d_implementation::direct;
	int64_t chunk_size = 0; // the size of each chunk to split the copy into, in bytes; 0 means no chunking

	copy_strategy() = default;
	copy_strategy(copy_type t) : type(t) {}
	copy_strategy(int64_t c) : chunk_size(c) {}
	copy_strategy(copy_type t, copy_properties p) : type(t), properties(p) {}
	copy_strategy(copy_type t, copy_properties p, d2d_implementation d) : type(t), properties(p), d2d(d) {}
	copy_strategy(copy_type t, copy_properties p, int64_t c) : type(t), properties(p), chunk_size(c) {}
	copy_strategy(copy_type t, copy_properties p, d2d_implementation d, int64_t c) : type(t), properties(p), d2d(d), chunk_size(c) {}

	constexpr bool operator==(const copy_strategy&) const = default;
	constexpr bool operator!=(const copy_strategy&) const = default;
};

// validate whether a given data layout is sound
bool is_valid(const data_layout& layout);

// validate whether a given copy spec is sound
bool is_valid(const copy_spec& spec);

// validate whether a given copy plan is sound
bool is_valid(const copy_plan& plan);

// validate whether a given copy set is sound
bool is_valid(const parallel_copy_set& set);

// Collapse densely packed dimensions into the unique normal form of the box (see "Normal form" in docs/design.md):
// a single contiguous run becomes a 1D layout, evenly spaced runs a 2D one, and anything else stays 3D.
// Two layouts describe the same box bytes and window exactly when their normal forms compare equal.
data_layout normalize(const data_layout& layout);

// Normalize each layout of a copy spec independently. Each side is read in its own packed order, so this preserves
// exactly what the spec copies.
copy_spec normalize(const copy_spec& spec);

// The widest element the copy kernels can use, i.e. the largest power of two up to 64 dividing the row extents,
// `d0_stride` and `d0_start_offset` of both sides and the shift between the two windows. Chunk boundaries are multiples
// of it, and a smaller `chunk_size` is raised to it. The bases are unknown while planning, so the kernels narrow the
// element further for a base aligned to less.
inline int64_t copy_alignment(const copy_spec& spec) {
	int64_t alignment = 64;
	int64_t source_width = spec.source_layout.d0_end_offset - spec.source_layout.d0_start_offset;
	int64_t target_width = spec.target_layout.d0_end_offset - spec.target_layout.d0_start_offset;
	int64_t shift = spec.source_layout.start - spec.target_layout.start;
	for(; alignment > 1; alignment >>= 1) {
		if(source_width % alignment == 0 && target_width % alignment == 0 && spec.source_layout.d0_stride % alignment == 0
		    && spec.target_layout.d0_stride % alignment == 0 && spec.source_layout.d0_start_offset % alignment == 0
		    && spec.target_layout.d0_start_offset % alignment == 0 && shift % alignment == 0) {
			break;
		}
	}
	return alignment;
}

// Invokes f(source_offset, target_offset, length) for each run of bytes that is contiguous on both sides of the copy, in
// packed order, with offsets relative to each side's allocation base. A run ends where a run of either side ends, so a
// reshaping copy yields the pieces both sides share. This is the iteration primitive for copies made of 1D copies.
// Both windows must have the same length, which is_valid(spec) checks.
template <typename F>
void for_each_copy_run(const copy_spec& spec, F&& f) {
	const auto& source = spec.source_layout;
	const auto& target = spec.target_layout;
	const int64_t length = source.window_length();
	const int64_t source_row = source.d0_end_offset - source.d0_start_offset;
	const int64_t target_row = target.d0_end_offset - target.d0_start_offset;
	int64_t run_source = 0;
	int64_t run_target = 0;
	int64_t run_length = 0;
	// the idea here is to build up the run length until there is an actual gap in the data, at which point f is being called
	for(int64_t i = 0; i < length;) {
		const int64_t source_offset = source.offset_at(source.start + i);
		const int64_t target_offset = target.offset_at(target.start + i);
		// the rest of the rows both sides are in, clipped to the window
		const int64_t take = std::min({source_row - (source.start + i) % source_row, target_row - (target.start + i) % target_row, length - i});
		if(run_length != 0 && (source_offset != run_source + run_length || target_offset != run_target + run_length)) {
			f(run_source, run_target, run_length);
			run_length = 0;
		}
		if(run_length == 0) {
			run_source = source_offset;
			run_target = target_offset;
		}
		run_length += take;
		i += take;
	}
	if(run_length != 0) { f(run_source, run_target, run_length); }
}

// apply chunking to the given copy spec if requested by the strategy
parallel_copy_set apply_chunking(const copy_spec& spec, const copy_strategy& strategy);

using staging_buffer_provider = std::function<staging_id(device_id, bool, int64_t)>;

class basic_staging_provider {
  public:
	staging_id operator()(device_id did, bool on_host, [[maybe_unused]] int64_t size) { return {on_host, did, next_staging_idx++}; }

  private:
	uint32_t next_staging_idx = 0;
};

// apply staging to the given spec if requested by the strategy
copy_plan apply_staging(const copy_spec& spec, const copy_strategy& strategy, const staging_buffer_provider& staging_provider);

// apply staging to each copy spec in the given parallel copy set if requested by the strategy
parallel_copy_set apply_staging(const parallel_copy_set& set, const copy_strategy& strategy, const staging_buffer_provider& staging_provider);

// apply the desired d2d implementation to the given copy plan
copy_plan apply_d2d_implementation(const copy_plan& plan, const d2d_implementation d2d, const staging_buffer_provider& staging_provider);

// apply the desired d2d implementation to the given parallel copy set (by applying it to each copy plan)
parallel_copy_set apply_d2d_implementation(const parallel_copy_set& set, const d2d_implementation d2d, const staging_buffer_provider& staging_provider);

// manifests the copy strategy on the normalized copy spec, applying chunking and staging as necessary
parallel_copy_set manifest_strategy(const copy_spec& spec, const copy_strategy& strategy, const staging_buffer_provider& staging_provider);

} // namespace copylib
