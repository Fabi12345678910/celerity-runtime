#include <copylib/core.hpp>
#include <copylib/support.hpp> // IWYU pragma: keep

#include <optional>

namespace copylib {

bool is_valid(const data_layout& layout) {
	return layout.d0_start_offset >= 0 && layout.d1_start_offset >= 0 && layout.d2_start_offset >= 0 && layout.start >= 0 && layout.d0_stride > 0
	       && layout.d1_stride > 0 && layout.d0_end_offset > layout.d0_start_offset && layout.d1_end_offset > layout.d1_start_offset
	       && layout.d2_end_offset > layout.d2_start_offset && layout.end > layout.start && layout.d0_end_offset <= layout.d0_stride
	       && layout.d1_end_offset <= layout.d1_stride && layout.end <= layout.total_bytes(); // the window lies inside the box
}

bool is_valid(const copy_spec& spec) {
	return is_valid(spec.source_layout) && is_valid(spec.target_layout) && spec.source_layout.window_length() == spec.target_layout.window_length();
}

bool is_valid(const copy_plan& plan) {
	for(const copy_spec& spec : plan) {
		if(!is_valid(spec)) { return false; }
	}
	return true;
}

bool is_valid(const parallel_copy_set& set) {
	for(const copy_plan& spec : set) {
		if(!is_valid(spec)) { return false; }
	}
	return true;
}

namespace {

	// A dimension of extent 1 is not merged by either collapse, there is nothing to merge. But its start offset and its
	// stride stop showing up in any byte offset, so they have to be pinned or the same bytes keep several encodings.
	void canonicalize(data_layout& layout) {
		if(layout.d2_end_offset - layout.d2_start_offset != 1) return;

		// one plane: fold its offset into d1, d1_stride is then free to be the smallest legal value
		const int64_t rows = layout.d2_start_offset * layout.d1_stride;
		layout.d1_start_offset += rows;
		layout.d1_end_offset += rows;
		layout.d2_start_offset = 0;
		layout.d2_end_offset = 1;
		layout.d1_stride = layout.d1_end_offset;

		if(layout.d1_end_offset - layout.d1_start_offset != 1) return;

		// one row as well: the box is a single run, so fold once more and d0_stride is free too
		const int64_t bytes = layout.d1_start_offset * layout.d0_stride;
		layout.d0_start_offset += bytes;
		layout.d0_end_offset += bytes;
		layout.d1_start_offset = 0;
		layout.d1_end_offset = 1;
		layout.d1_stride = 1;
		layout.d0_stride = layout.d0_end_offset;
	}

	bool collapse_d1_onto_d0(data_layout& layout) {
		if(!layout.d1_contiguous()) return false;

		data_layout old_layout(layout);
		// collapse d1 onto d0
		layout.d0_stride = old_layout.d0_stride * old_layout.d1_stride;
		layout.d0_start_offset = old_layout.d1_start_offset * old_layout.d0_stride;
		layout.d0_end_offset = old_layout.d1_end_offset * old_layout.d0_stride;

		layout.d1_start_offset = old_layout.d2_start_offset;
		layout.d1_end_offset = old_layout.d2_end_offset;

		// see collapse_d2_onto_d1 for explanation
		layout.d1_stride = layout.d1_end_offset;
		layout.d2_start_offset = 0;
		layout.d2_end_offset = 1;
		return true;
	}

	bool collapse_d2_onto_d1(data_layout& layout) {
		if(!((layout.d1_start_offset == 0 && layout.d1_end_offset == layout.d1_stride))) return false;

		data_layout old_layout(layout);

		layout.d1_start_offset = old_layout.d2_start_offset * old_layout.d1_stride;
		layout.d1_end_offset = old_layout.d2_end_offset * old_layout.d1_stride;

		// after collapse, d1_stride is not used at retrieving any offsets anymore
		// but the is_valid check will test whether d1 extends d1_stride, so we have to adjust d1_stride to contain d1
		layout.d1_stride = layout.d1_end_offset;
		layout.d2_start_offset = 0;
		layout.d2_end_offset = 1;
		return true;
	}

} // namespace

data_layout normalize(const data_layout& layout) {
	data_layout new_layout(layout);
	collapse_d2_onto_d1(new_layout);
	collapse_d1_onto_d0(new_layout);
	canonicalize(new_layout);
	return new_layout;
}

copy_spec normalize(const copy_spec& spec) {
	return {spec.source_device, normalize(spec.source_layout), spec.target_device, normalize(spec.target_layout), spec.properties};
}

parallel_copy_set apply_chunking(const copy_spec& spec, const copy_strategy& strategy) {
	COPYLIB_ENSURE(is_valid(spec), "Invalid copy specification, cannot chunk: {}", spec);
	if(strategy.chunk_size == 0) { return {{spec}}; }
	parallel_copy_set copy_set;

	int64_t alignment = copy_alignment(spec);

	int64_t used_chunk_size = std::max(alignment, (strategy.chunk_size / alignment) * alignment);

	int64_t source_start = spec.source_layout.start;
	int64_t target_start = spec.target_layout.start;

	while(source_start < spec.source_layout.end) {
		int64_t source_end = std::min((source_start / used_chunk_size + 1) * used_chunk_size, spec.source_layout.end);
		int64_t target_end = target_start + (source_end - source_start);


		copy_set.push_back({{spec.source_device, spec.source_layout.with_window(source_start, source_end), spec.target_device,
		    spec.target_layout.with_window(target_start, target_end), spec.properties}});

		source_start = source_end;
		target_start = target_end;
	}

	return copy_set;
}

copy_plan apply_staging(const copy_spec& spec, const copy_strategy& strategy, const staging_buffer_provider& staging_provider) {
	COPYLIB_ENSURE(is_valid(spec), "Invalid copy specification, cannot stage: {}", spec);
	const auto proper_spec = spec.with_properties(strategy.properties);

	if(spec.source_device == device_id::host && spec.target_device == device_id::host) { return {proper_spec}; }

	if(strategy.type == copy_type::direct) { return {proper_spec}; }
	if(strategy.type != copy_type::staged) {
		COPYLIB_ERROR("Unknown copy strategy type: {}", strategy.type);
		return {proper_spec};
	}

	// if we are looking at a contiguous copy, we don't need to stage, but we need to normalize the layouts
	if(spec.source_layout.is_window_contiguous() && spec.target_layout.is_window_contiguous()) { return {normalize(proper_spec)}; }
	// if the source is not unit stride, we need to stage the source
	std::optional<copy_spec> source_staging_copy;
	if(!spec.source_layout.is_window_contiguous()) {
		const auto device_id_for_staging =
		    spec.source_device != device_id::host ? spec.source_device : (spec.target_device != device_id::host ? spec.target_device : device_id::d0);
		const auto source_staging_buffer = staging_provider(device_id_for_staging, spec.source_device == device_id::host, spec.source_layout.window_length());
		const data_layout staged_source_layout = data_layout(source_staging_buffer, 0, spec.source_layout.window_length());
		source_staging_copy.emplace(spec.source_device, spec.source_layout, spec.source_device, staged_source_layout, strategy.properties);
	}

	// if the target is not unit stride, we need to unstage the target
	std::optional<copy_spec> target_unstaging_copy;
	if(!spec.target_layout.is_window_contiguous()) {
		const auto device_id_for_staging =
		    spec.target_device != device_id::host ? spec.target_device : (spec.source_device != device_id::host ? spec.source_device : device_id::d0);
		const auto target_staging_buffer = staging_provider(device_id_for_staging, spec.target_device == device_id::host, spec.target_layout.window_length());
		const data_layout staged_target_layout = data_layout(target_staging_buffer, 0, spec.target_layout.window_length());
		target_unstaging_copy.emplace(spec.target_device, staged_target_layout, spec.target_device, spec.target_layout, strategy.properties);
	}

	// now we can build the copy plan
	copy_plan plan;
	if(source_staging_copy.has_value() && target_unstaging_copy.has_value()) {
		const auto& src = source_staging_copy.value();
		const auto& tgt = target_unstaging_copy.value();
		plan.push_back(src);
		plan.emplace_back(src.source_device, src.target_layout, tgt.target_device, tgt.source_layout, strategy.properties);
		plan.push_back(tgt);
	} else if(source_staging_copy.has_value()) {
		const auto& src = source_staging_copy.value();
		plan.push_back(src);
		plan.emplace_back(src.target_device, src.target_layout, spec.target_device, spec.target_layout, strategy.properties);
	} else if(target_unstaging_copy.has_value()) {
		const auto& tgt = target_unstaging_copy.value();
		plan.emplace_back(spec.source_device, spec.source_layout, tgt.source_device, tgt.source_layout, strategy.properties);
		plan.push_back(tgt);
	} else {
		COPYLIB_ERROR("Something strange is afoot when staging: {}", spec);
	}
	return plan;
}
parallel_copy_set apply_staging(const parallel_copy_set& set, const copy_strategy& strategy, const staging_buffer_provider& staging_provider) {
	parallel_copy_set copies;
	for(const auto& copy : set) {
		COPYLIB_ENSURE(copy.size() == 1, "Cannot stage a copy set with plans consisting of more than one copy (plan: {})", copy);
		copies.push_back(apply_staging(copy.front(), strategy, staging_provider));
	}
	return copies;
}

copy_plan apply_d2d_implementation(const copy_plan& plan, const d2d_implementation d2d, const staging_buffer_provider& staging_provider) {
	COPYLIB_ENSURE(is_valid(plan), "Invalid copy plan, cannot apply d2d implementation: {}", plan);
	if(d2d == d2d_implementation::direct) { return plan; }
	// we need to change any copies that go from a device to another device
	copy_plan new_plan;
	for(const auto& spec : plan) {
		if(spec.source_device == spec.target_device || spec.source_device == device_id::host || spec.target_device == device_id::host) {
			new_plan.push_back(spec);
		} else {
			// host staging buffers are packed 1D and sized to the window, where the 2D version mirrored the source's fragment layout
			switch(d2d) {
			case d2d_implementation::host_staging_at_source: {
				const auto staging_buffer = staging_provider(spec.source_device, true, spec.source_layout.window_length());
				const data_layout staged_layout = {staging_buffer, 0, spec.source_layout.window_length()};
				new_plan.emplace_back(spec.source_device, spec.source_layout, device_id::host, staged_layout, spec.properties);
				new_plan.emplace_back(device_id::host, staged_layout, spec.target_device, spec.target_layout, spec.properties);
				break;
			}
			case d2d_implementation::host_staging_at_target: {
				const auto staging_buffer = staging_provider(spec.target_device, true, spec.source_layout.window_length());
				const data_layout staged_layout = {staging_buffer, 0, spec.source_layout.window_length()};
				new_plan.emplace_back(spec.source_device, spec.source_layout, device_id::host, staged_layout, spec.properties);
				new_plan.emplace_back(device_id::host, staged_layout, spec.target_device, spec.target_layout, spec.properties);
				break;
			}
			case d2d_implementation::host_staging_at_both: {
				const auto source_staging_buffer = staging_provider(spec.source_device, true, spec.source_layout.window_length());
				const data_layout staged_source_layout = {source_staging_buffer, 0, spec.source_layout.window_length()};
				new_plan.emplace_back(spec.source_device, spec.source_layout, device_id::host, staged_source_layout, spec.properties);
				const auto target_staging_buffer = staging_provider(spec.target_device, true, spec.target_layout.window_length());
				const data_layout staged_target_layout = {target_staging_buffer, 0, spec.target_layout.window_length()};
				new_plan.emplace_back(device_id::host, staged_source_layout, device_id::host, staged_target_layout, spec.properties);
				new_plan.emplace_back(device_id::host, staged_target_layout, spec.target_device, spec.target_layout, spec.properties);
				break;
			}
			default: COPYLIB_ERROR("Unknown d2d implementation: {}", d2d);
			}
		}
	}
	return new_plan;
}

parallel_copy_set apply_d2d_implementation(const parallel_copy_set& copy_set, const d2d_implementation d2d, const staging_buffer_provider& staging_provider) {
	parallel_copy_set ret;
	for(const auto& plan : copy_set) {
		ret.push_back(apply_d2d_implementation(plan, d2d, staging_provider));
	}
	return ret;
}

parallel_copy_set manifest_strategy(const copy_spec& spec, const copy_strategy& strategy, const staging_buffer_provider& staging_provider) {
	// checked before normalizing, which assumes a valid spec
	COPYLIB_ENSURE(is_valid(spec), "Invalid copy specification, cannot manifest: {}", spec);
	const auto normalized_spec = normalize(spec);
	const auto chunked_copies = apply_chunking(normalized_spec, strategy);
	const auto staged_copies = apply_staging(chunked_copies, strategy, staging_provider);
	const auto finalized_copies = apply_d2d_implementation(staged_copies, strategy.d2d, staging_provider);
	return finalized_copies;
}

} // namespace copylib
