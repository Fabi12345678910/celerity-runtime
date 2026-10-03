#include <copylib/backend.hpp>

#include <copylib/support.hpp> // IWYU pragma: keep

#include <cstdlib>
#include <limits>

namespace copylib::detail {

// One side of a copy counted in elements of the kernel's type, which divides every extent, stride and offset.
template <typename IdxType>
struct element_layout {
	IdxType row_extent;   // elements per row of the box
	IdxType rows;         // rows per plane of the box
	IdxType d0_stride;    // elements per row of the allocation
	IdxType plane_stride; // elements per plane of the allocation
	IdxType d0_start_offset;
	IdxType d1_start_offset;
	IdxType d2_start_offset;

	element_layout(const data_layout& l, int64_t elem_size)
	    : row_extent((l.d0_end_offset - l.d0_start_offset) / elem_size), rows(l.d1_end_offset - l.d1_start_offset), d0_stride(l.d0_stride / elem_size),
	      plane_stride(l.d1_stride * l.d0_stride / elem_size), d0_start_offset(l.d0_start_offset / elem_size), d1_start_offset(l.d1_start_offset),
	      d2_start_offset(l.d2_start_offset) {}

	// data_layout::offset_at in elements: the allocation offset of the element `packed` elements into the box
	IdxType offset_at(IdxType packed) const {
		const IdxType row = packed / row_extent;
		return (row / rows + d2_start_offset) * plane_stride + (row % rows + d1_start_offset) * d0_stride + packed % row_extent + d0_start_offset;
	}
};

template <typename T, typename IdxType>
sycl::event copy_with_kernel_impl(sycl::queue& q, const copy_spec& spec, IdxType preferred_wg_size) {
	const T* src = reinterpret_cast<const T*>(spec.source_layout.base_ptr());
	T* tgt = reinterpret_cast<T*>(spec.target_layout.base_ptr());
	const element_layout<IdxType> src_layout(spec.source_layout, sizeof(T));
	const element_layout<IdxType> tgt_layout(spec.target_layout, sizeof(T));
	const IdxType src_start = spec.source_layout.start / sizeof(T);
	const IdxType tgt_start = spec.target_layout.start / sizeof(T);

	// the global range is rounded up to whole work groups, so the work items past the window return right away
	const IdxType extent = spec.source_layout.window_length() / sizeof(T);
	const size_t wg_size = preferred_wg_size;
	const IdxType per_item = std::getenv("COPYLIB_ELEMS_PER_ITEM") ? std::atoi(std::getenv("COPYLIB_ELEMS_PER_ITEM")) : 1;
	const sycl::nd_range<1> ndr{(static_cast<size_t>(extent) + wg_size * per_item - 1) / (wg_size * per_item) * wg_size, wg_size};
	return q.parallel_for(ndr, [=](sycl::nd_item<1> idx) {
		for(IdxType n = 0, i = idx.get_group(0) * wg_size * per_item + idx.get_local_id(0); n < per_item && i < extent; n++, i += wg_size) {
			tgt[tgt_layout.offset_at(tgt_start + i)] = src[src_layout.offset_at(src_start + i)];
		}
	});
}

template <typename T>
sycl::event copy_with_kernel_impl(sycl::queue& q, const copy_spec& spec, int32_t preferred_wg_size) {
	// int32 indices need every packed index and every allocation offset of both windows to fit; both grow with the index
	const int64_t max = std::numeric_limits<int32_t>::max();
	const auto& src = spec.source_layout;
	const auto& tgt = spec.target_layout;
	if(src.end / static_cast<int64_t>(sizeof(T)) < max && tgt.end / static_cast<int64_t>(sizeof(T)) < max
	    && src.offset_at(src.end - 1) / static_cast<int64_t>(sizeof(T)) < max && tgt.offset_at(tgt.end - 1) / static_cast<int64_t>(sizeof(T)) < max) {
		return copy_with_kernel_impl<T, int32_t>(q, spec, preferred_wg_size);
	} else {
		return copy_with_kernel_impl<T, int64_t>(q, spec, preferred_wg_size);
	}
}

sycl::event copy_with_kernel(sycl::queue& q, const copy_spec& spec, int32_t preferred_wg_size) {
	// the widest element that tiles both windows: copy_alignment covers the rows, strides and offsets of both sides and the
	// shift between the windows, which leaves the window's own start and length, as chunks may start and end unaligned, and
	// both bases, which planning does not know
	int64_t elem_size = std::min<int64_t>(copy_alignment(spec), std::getenv("COPYLIB_MAX_ELEM_SIZE") ? std::atoi(std::getenv("COPYLIB_MAX_ELEM_SIZE")) : 64);
	while(spec.source_layout.start % elem_size != 0 || spec.source_layout.window_length() % elem_size != 0 || spec.source_layout.base % elem_size != 0
	      || spec.target_layout.base % elem_size != 0) {
		elem_size /= 2;
	}
	switch(elem_size) {
	case 64: return copy_with_kernel_impl<sycl::int16>(q, spec, preferred_wg_size);
	case 32: return copy_with_kernel_impl<sycl::int8>(q, spec, preferred_wg_size);
	case 16: return copy_with_kernel_impl<sycl::int4>(q, spec, preferred_wg_size);
	case 8: return copy_with_kernel_impl<sycl::int2>(q, spec, preferred_wg_size);
	case 4: return copy_with_kernel_impl<int32_t>(q, spec, preferred_wg_size);
	case 2: return copy_with_kernel_impl<int16_t>(q, spec, preferred_wg_size);
	default: return copy_with_kernel_impl<int8_t>(q, spec, preferred_wg_size);
	}
}

} // namespace copylib::detail
