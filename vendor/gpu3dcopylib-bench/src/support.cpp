#include <copylib/support.hpp>

#include <string>

// Ported from the 2D library's copylib_support.hpp, whose bodies were inline in the header.
// Everything that does not mention the layout fields is unchanged.

namespace std {

size_t hash<copylib::data_layout>::operator()(const copylib::data_layout& layout) const {
	return copylib::utils::hash_args(layout.base, layout.d0_stride, layout.d1_stride, layout.d0_start_offset, layout.d1_start_offset, layout.d2_start_offset,
	    layout.d0_end_offset, layout.d1_end_offset, layout.d2_end_offset, layout.start, layout.end);
}

size_t hash<copylib::copy_properties>::operator()(const copylib::copy_properties& prop) const { //
	return hash<int>{}(static_cast<int>(prop));
}

size_t hash<copylib::copy_spec>::operator()(const copylib::copy_spec& spec) const {
	return copylib::utils::hash_args(spec.source_device, spec.source_layout, spec.target_device, spec.target_layout, spec.properties);
}

size_t hash<copylib::copy_plan>::operator()(const copylib::copy_plan& plan) const {
	size_t val = 0;
	for(const auto& spec : plan) {
		copylib::utils::hash_combine(val, hash<copylib::copy_spec>{}(spec));
	}
	return val;
}

size_t hash<copylib::copy_type>::operator()(const copylib::copy_type& type) const { return hash<int>{}(static_cast<int>(type)); }

size_t hash<copylib::d2d_implementation>::operator()(const copylib::d2d_implementation& impl) const { return hash<int>{}(static_cast<int>(impl)); }

size_t hash<copylib::copy_strategy>::operator()(const copylib::copy_strategy& strat) const {
	return copylib::utils::hash_args(strat.type, strat.properties, strat.d2d, strat.chunk_size);
}

} // namespace std

#ifdef COPYLIB_USE_FMT
namespace fmt {
#else
namespace std {
#endif

format_context::iterator formatter<copylib::device_id>::format(const copylib::device_id& p, format_context& ctx) const {
	if(p == copylib::device_id::host) { return formatter<std::string>::format("host", ctx); }
	return formatter<std::string>::format(copylib::utils::format("d{}", static_cast<int>(p)), ctx);
}

format_context::iterator formatter<copylib::staging_id>::format(const copylib::staging_id& p, format_context& ctx) const {
	COPYLIB_ENSURE(p.is_staging_id == copylib::staging_id::staging_id_flag, "Invalid staging id");
	return formatter<std::string>::format(copylib::utils::format("S({}, {}{})", p.index, p.did, p.on_host ? "host" : ""), ctx);
}

format_context::iterator formatter<copylib::data_layout>::format(const copylib::data_layout& p, format_context& ctx) const {
	const std::string addr =
	    p.is_unplaced_staging() ? copylib::utils::format("{}", p.staging) : copylib::utils::format("{:p}", reinterpret_cast<void*>(p.base));
	return formatter<std::string>::format(
	    copylib::utils::format("{{{} [{}x{}] d0:[{},{}) d1:[{},{}) d2:[{},{}) win:[{},{})}}", addr, p.d0_stride, p.d1_stride, p.d0_start_offset,
	        p.d0_end_offset, p.d1_start_offset, p.d1_end_offset, p.d2_start_offset, p.d2_end_offset, p.start, p.end),
	    ctx);
}

format_context::iterator formatter<copylib::copy_properties>::format(const copylib::copy_properties& p, format_context& ctx) const {
	std::string result;
	if(p & copylib::copy_properties::use_kernel) { result += "use_kernel"; }
	return formatter<std::string>::format(result, ctx);
}

format_context::iterator formatter<copylib::copy_spec>::format(const copylib::copy_spec& p, format_context& ctx) const {
	using namespace std::string_literals;
	auto prop_string = ""s;
	if(p.properties != copylib::copy_properties::none) { prop_string = copylib::utils::format(" ({})", p.properties); }
	return formatter<std::string>::format(
	    copylib::utils::format("copy({}{}, {}{}{})", p.source_device, p.source_layout, p.target_device, p.target_layout, prop_string), ctx);
}

format_context::iterator formatter<copylib::copy_type>::format(const copylib::copy_type& p, format_context& ctx) const {
	switch(p) {
	case copylib::copy_type::direct: return formatter<std::string>::format("direct", ctx);
	case copylib::copy_type::staged: return formatter<std::string>::format("staged", ctx);
	default: COPYLIB_ERROR("Unknown copy type {}", static_cast<int>(p));
	}
}

format_context::iterator formatter<copylib::d2d_implementation>::format(const copylib::d2d_implementation& p, format_context& ctx) const {
	switch(p) {
	case copylib::d2d_implementation::direct: return formatter<std::string>::format("direct", ctx);
	case copylib::d2d_implementation::host_staging_at_source: return formatter<std::string>::format("host_staging_at_source", ctx);
	case copylib::d2d_implementation::host_staging_at_target: return formatter<std::string>::format("host_staging_at_target", ctx);
	case copylib::d2d_implementation::host_staging_at_both: return formatter<std::string>::format("host_staging_at_both", ctx);
	default: COPYLIB_ERROR("Unknown d2d implementation {}", static_cast<int>(p));
	}
}

format_context::iterator formatter<copylib::copy_strategy>::format(const copylib::copy_strategy& p, format_context& ctx) const {
	return formatter<std::string>::format(copylib::utils::format("strategy({}, {}, d2d:{}, chunk:{})", p.type, p.properties, p.d2d, p.chunk_size), ctx);
}

format_context::iterator formatter<copylib::copy_plan>::format(const copylib::copy_plan& p, format_context& ctx) const {
	ctx.advance_to(format_to(ctx.out(), "["));
	for(size_t i = 0; i < p.size(); i++) {
		const auto& spec = p[i];
		ctx.advance_to(formatter<copylib::copy_spec>{}.format(spec, ctx));
		if(i < p.size() - 1) ctx.advance_to(format_to(ctx.out(), ", "));
	}
	return format_to(ctx.out(), "]");
}

format_context::iterator formatter<copylib::parallel_copy_set>::format(const copylib::parallel_copy_set& p, format_context& ctx) const {
	ctx.advance_to(format_to(ctx.out(), "{{"));
	auto it = p.cbegin();
	for(size_t i = 0; i < p.size(); i++) {
		const auto& plan = *it++;
		ctx.advance_to(formatter<copylib::copy_plan>{}.format(plan, ctx));
		if(i < p.size() - 1) ctx.advance_to(format_to(ctx.out(), ", "));
	}
	return format_to(ctx.out(), "}}");
}

} // namespace fmt // std

#ifdef COPYLIB_USE_FMT
#define format_to(_a, _b, _c) fmt::format_to(_a, _b, _c)
#endif

namespace std {

#define COPYLIB_OSTREAM_FOR(__type)                                                                                                                            \
	std::ostream& operator<<(std::ostream& os, const copylib::__type& p) {                                                                                     \
		format_to(std::ostreambuf_iterator<char>(os), "{}", p);                                                                                                \
		return os;                                                                                                                                             \
	}

COPYLIB_OSTREAM_FOR(device_id)
COPYLIB_OSTREAM_FOR(staging_id)
COPYLIB_OSTREAM_FOR(data_layout)
COPYLIB_OSTREAM_FOR(copy_properties)
COPYLIB_OSTREAM_FOR(copy_spec)
COPYLIB_OSTREAM_FOR(copy_type)
COPYLIB_OSTREAM_FOR(d2d_implementation)
COPYLIB_OSTREAM_FOR(copy_strategy)
COPYLIB_OSTREAM_FOR(copy_plan)
COPYLIB_OSTREAM_FOR(parallel_copy_set)

#undef COPYLIB_OSTREAM_FOR

} // namespace std
