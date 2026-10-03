#include "test_utils.hpp"

#include <copylib/core.hpp>
#include <copylib/support.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_tostring.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <functional>
#include <sstream>
#include <string>
#include <vector>

// Layer 8: hashing and formatting, ported from the 2D suite.
//
// Hashes: equal values hash equal, and changing any single field changes the hash. The window is part of a layout's
// identity, so a hash ignoring start and end would give every chunk of a spec the same hash. Every "equal" check is
// paired with a "different" one, since a hash returning the same value for everything passes every "equal" check.
//
// Formatting: the enum-like types keep their exact 2D output. The 3D data_layout format is not fixed yet, so only its
// properties are checked: it is not empty, it shows the base address or the staging id, and different layouts print
// differently.

using namespace copylib;
using namespace copylib_testing;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::EndsWith;
using Catch::Matchers::StartsWith;
namespace ref = copylib_testing::reference_box;

namespace {

template <typename T>
size_t hash_of(const T& value) {
	return std::hash<T>{}(value);
}

// the reference box with one field changed per entry
std::vector<data_layout> single_field_changes() {
	std::vector<data_layout> changed(11, ref::fields());
	changed[0].base += 0x100;
	changed[1].d0_stride += 8;
	changed[2].d1_stride += 1;
	changed[3].d0_start_offset -= 8;
	changed[4].d1_start_offset -= 1;
	changed[5].d2_start_offset -= 1;
	changed[6].d0_end_offset += 8;
	changed[7].d1_end_offset += 1;
	changed[8].d2_end_offset += 1;
	changed[9].start += 8;
	changed[10].end -= 8;
	return changed;
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// Hashing

TEST_CASE("layouts hash by all of their fields, window included", "[support][hash]") {
	CHECK(hash_of(ref::fields()) == hash_of(ref::fields()));

	const char* field_names[] = {"base", "d0_stride", "d1_stride", "d0_start_offset", "d1_start_offset", "d2_start_offset", "d0_end_offset", "d1_end_offset",
	    "d2_end_offset", "start", "end"};
	const auto changed = single_field_changes();
	for(size_t i = 0; i < changed.size(); ++i) {
		INFO("changed field: " << field_names[i]);
		CHECK(hash_of(changed[i]) != hash_of(ref::fields()));
	}
}

TEST_CASE("chunks of one spec hash differently", "[support][hash][window]") {
	const auto layout = ref::fields();
	CHECK(hash_of(with_window_fields(layout, 0, 96)) != hash_of(with_window_fields(layout, 96, 192)));
	CHECK(hash_of(with_window_fields(layout, 0, 96)) == hash_of(with_window_fields(layout, 0, 96)));
}

TEST_CASE("normalized encodings of the same bytes hash equal", "[support][hash][normalization]") {
	// canonical encoding is what makes this hold: after normalize, the same bytes have the same fields
	const auto in_allocation = normalize(shapes::single_row());
	const auto built_1d = normalize(normal_form::one_run(ref::base, 16, 24));
	REQUIRE(same_fields(in_allocation, built_1d));

	CHECK(hash_of(in_allocation) == hash_of(built_1d));
	CHECK(hash_of(in_allocation) != hash_of(normalize(shapes::whole_allocation())));
}

TEST_CASE("copy specs hash by devices, layouts and properties", "[support][hash]") {
	const auto spec = ref::spec();
	CHECK(hash_of(spec) == hash_of(ref::spec()));

	auto other_source_device = spec;
	other_source_device.source_device = device_id::d2;
	auto other_target_device = spec;
	other_target_device.target_device = device_id::host;
	auto other_layout = spec;
	other_layout.target_layout = with_window_fields(spec.target_layout, 0, 96);
	auto other_properties = spec;
	other_properties.properties = copy_properties::use_kernel;

	CHECK(hash_of(other_source_device) != hash_of(spec));
	CHECK(hash_of(other_target_device) != hash_of(spec));
	CHECK(hash_of(other_layout) != hash_of(spec));
	CHECK(hash_of(other_properties) != hash_of(spec));
}

TEST_CASE("copy plans hash by their steps in order", "[support][hash]") {
	const auto a = ref::spec();
	const auto b = ref::spec().with_properties(copy_properties::use_kernel);

	CHECK(hash_of(copy_plan{a, b}) == hash_of(copy_plan{a, b}));
	CHECK(hash_of(copy_plan{a}) != hash_of(copy_plan{a, a}));
	// plans are sequences and compare in order, so their hashes must too
	CHECK(hash_of(copy_plan{a, b}) != hash_of(copy_plan{b, a}));
}

TEST_CASE("strategies hash by all of their fields", "[support][hash]") {
	const auto base = strategy_from_fields(copy_type::staged, copy_properties::none, d2d_implementation::direct, 64);
	CHECK(hash_of(base) == hash_of(strategy_from_fields(copy_type::staged, copy_properties::none, d2d_implementation::direct, 64)));

	CHECK(hash_of(strategy_from_fields(copy_type::direct, copy_properties::none, d2d_implementation::direct, 64)) != hash_of(base));
	CHECK(hash_of(strategy_from_fields(copy_type::staged, copy_properties::use_kernel, d2d_implementation::direct, 64)) != hash_of(base));
	CHECK(hash_of(strategy_from_fields(copy_type::staged, copy_properties::none, d2d_implementation::host_staging_at_both, 64)) != hash_of(base));
	CHECK(hash_of(strategy_from_fields(copy_type::staged, copy_properties::none, d2d_implementation::direct, 128)) != hash_of(base));
}

TEST_CASE("enum values hash distinctly", "[support][hash]") {
	CHECK(hash_of(copy_properties::none) == hash_of(copy_properties::none));
	CHECK(hash_of(copy_properties::none) != hash_of(copy_properties::use_kernel));
	CHECK(hash_of(copy_type::direct) != hash_of(copy_type::staged));
	CHECK(hash_of(d2d_implementation::host_staging_at_source) != hash_of(d2d_implementation::host_staging_at_target));
}

// ---------------------------------------------------------------------------------------------------------------------
// Formatting

TEST_CASE("enum-like types keep their 2D format", "[support][format]") {
	CHECK(utils::format("{}", device_id::host) == "host");
	CHECK(utils::format("{}", device_id::d0) == "d0");
	CHECK(utils::format("{}", device_id::d5) == "d5");

	CHECK(utils::format("{}", copy_type::direct) == "direct");
	CHECK(utils::format("{}", copy_type::staged) == "staged");

	CHECK(utils::format("{}", d2d_implementation::direct) == "direct");
	CHECK(utils::format("{}", d2d_implementation::host_staging_at_source) == "host_staging_at_source");
	CHECK(utils::format("{}", d2d_implementation::host_staging_at_target) == "host_staging_at_target");
	CHECK(utils::format("{}", d2d_implementation::host_staging_at_both) == "host_staging_at_both");

	CHECK(utils::format("{}", copy_properties::use_kernel) == "use_kernel");

	CHECK(utils::format("{}", staging_id_from_fields(true, device_id::d0, 42)) == "S(42, d0host)");
	CHECK(utils::format("{}", staging_id_from_fields(false, device_id::d1, 0)) == "S(0, d1)");

	CHECK(utils::format("{}", strategy_from_fields(copy_type::direct, copy_properties::use_kernel, d2d_implementation::direct, 256))
	      == "strategy(direct, use_kernel, d2d:direct, chunk:256)");
}

TEST_CASE("a layout prints its address and tells layouts apart", "[support][format]") {
	const auto layout = ref::fields();
	const auto printed = utils::format("{}", layout);

	CHECK_FALSE(printed.empty());
	CHECK_THAT(printed, ContainsSubstring("0x10000"));

	// different windows, strides and bases must be visible, or two chunks of a failing plan look identical
	for(const auto& other : single_field_changes()) {
		CHECK(utils::format("{}", other) != printed);
	}
}

TEST_CASE("an unplaced staging layout prints its staging id instead of an address", "[support][format][staging]") {
	const auto id = staging_id_from_fields(true, device_id::d1, 7);
	const auto printed = utils::format("{}", staging_layout_from_fields(id, 288));

	CHECK_FALSE(printed.empty());
	CHECK_THAT(printed, ContainsSubstring(utils::format("{}", id)));
}

TEST_CASE("composite types print their parts in order", "[support][format]") {
	const auto a = ref::spec();
	const auto b = spec_from_fields(device_id::d1, ref::fields(0x80000), device_id::host, ref::fields(0x90000)).with_properties(copy_properties::use_kernel);

	const auto spec_text = utils::format("{}", b);
	CHECK_THAT(spec_text, ContainsSubstring(utils::format("{}", b.source_layout)));
	CHECK_THAT(spec_text, ContainsSubstring(utils::format("{}", b.target_layout)));
	CHECK_THAT(spec_text, ContainsSubstring("host"));
	CHECK_THAT(spec_text, ContainsSubstring("use_kernel"));
	CHECK(spec_text.find(utils::format("{}", b.source_layout)) < spec_text.find(utils::format("{}", b.target_layout)));

	const auto plan_text = utils::format("{}", copy_plan{a, b});
	CHECK_THAT(plan_text, StartsWith("["));
	CHECK_THAT(plan_text, EndsWith("]"));
	CHECK(plan_text.find(utils::format("{}", a)) < plan_text.find(utils::format("{}", b)));

	const auto set_text = utils::format("{}", parallel_copy_set{copy_plan{a}, copy_plan{a, b}});
	CHECK_THAT(set_text, StartsWith("{"));
	CHECK_THAT(set_text, EndsWith("}"));
	CHECK_THAT(set_text, ContainsSubstring(utils::format("{}", copy_plan{a, b})));
}

TEST_CASE("ostream and Catch2 print exactly what the formatter prints", "[support][format]") {
	// Catch2 prints failed comparisons through these, so they are what a failing test shows
	const auto spec = ref::spec();
	const auto formatted = utils::format("{}", spec);
	REQUIRE_FALSE(formatted.empty());

	std::ostringstream os;
	os << spec;
	CHECK(os.str() == formatted);
	CHECK(Catch::Detail::stringify(spec) == formatted);
	CHECK(Catch::Detail::stringify(spec.source_layout) == utils::format("{}", spec.source_layout));
}
