#pragma once

#include "core.hpp"

#include "utils.hpp"

#include <ostream>

// this file contains the implementation of exceedingly boring template functions which really shouldn't
// be necessary at all in a modern programming language. Reflection when?

// make types hashable
namespace std {
template <>
struct hash<copylib::data_layout> {
	size_t operator()(const copylib::data_layout& layout) const;
};
template <>
struct hash<copylib::copy_properties> {
	size_t operator()(const copylib::copy_properties& prop) const;
};
template <>
struct hash<copylib::copy_spec> {
	size_t operator()(const copylib::copy_spec& spec) const;
};
template <>
struct hash<copylib::copy_plan> {
	size_t operator()(const copylib::copy_plan& plan) const;
};
template <>
struct hash<copylib::copy_type> {
	size_t operator()(const copylib::copy_type& type) const;
};
template <>
struct hash<copylib::d2d_implementation> {
	size_t operator()(const copylib::d2d_implementation& impl) const;
};
template <>
struct hash<copylib::copy_strategy> {
	size_t operator()(const copylib::copy_strategy& strat) const;
};
} // namespace std

#ifdef COPYLIB_USE_FMT
namespace fmt {
#else
namespace std {
#endif

// make types format-printable (and ostream for Catch2)
template <>
struct formatter<copylib::device_id> : formatter<std::string> {
	format_context::iterator format(const copylib::device_id& p, format_context& ctx) const;
};
template <>
struct formatter<copylib::staging_id> : formatter<std::string> {
	format_context::iterator format(const copylib::staging_id& p, format_context& ctx) const;
};
template <>
struct formatter<copylib::data_layout> : formatter<std::string> {
	format_context::iterator format(const copylib::data_layout& p, format_context& ctx) const;
};
template <>
struct formatter<copylib::copy_properties> : formatter<std::string> {
	format_context::iterator format(const copylib::copy_properties& p, format_context& ctx) const;
};
template <>
struct formatter<copylib::copy_spec> : formatter<std::string> {
	format_context::iterator format(const copylib::copy_spec& p, format_context& ctx) const;
};
template <>
struct formatter<copylib::copy_type> : formatter<std::string> {
	format_context::iterator format(const copylib::copy_type& p, format_context& ctx) const;
};
template <>
struct formatter<copylib::d2d_implementation> : formatter<std::string> {
	format_context::iterator format(const copylib::d2d_implementation& p, format_context& ctx) const;
};
template <>
struct formatter<copylib::copy_strategy> : formatter<std::string> {
	format_context::iterator format(const copylib::copy_strategy& p, format_context& ctx) const;
};
template <>
struct formatter<copylib::copy_plan> : formatter<std::string> {
	format_context::iterator format(const copylib::copy_plan& p, format_context& ctx) const;
};
template <>
struct formatter<copylib::parallel_copy_set> : formatter<std::string> {
	format_context::iterator format(const copylib::parallel_copy_set& p, format_context& ctx) const;
};
} // namespace fmt // std

namespace std {
ostream& operator<<(ostream& os, const copylib::device_id& p);
ostream& operator<<(ostream& os, const copylib::staging_id& p);
ostream& operator<<(ostream& os, const copylib::data_layout& p);
ostream& operator<<(ostream& os, const copylib::copy_properties& p);
ostream& operator<<(ostream& os, const copylib::copy_spec& p);
ostream& operator<<(ostream& os, const copylib::copy_type& p);
ostream& operator<<(ostream& os, const copylib::d2d_implementation& p);
ostream& operator<<(ostream& os, const copylib::copy_strategy& p);
ostream& operator<<(ostream& os, const copylib::copy_plan& p);
ostream& operator<<(ostream& os, const copylib::parallel_copy_set& p);
} // namespace std
