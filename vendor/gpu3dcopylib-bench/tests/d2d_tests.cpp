#include "test_utils.hpp"

#include <copylib/core.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <utility>
#include <vector>

// Layer 6: apply_d2d_implementation() routes copies between two different devices through pinned host buffers.
//
// Carried over from the 2D library: only steps between two different devices are rewritten; host_staging_at_source
// and _at_target place one host buffer near that device, host_staging_at_both one near each with a host-to-host copy
// between them; each rewritten step keeps the properties of the step it replaces. New in 3D (design.md): the host
// buffers are packed 1D, sized to the window.

using namespace copylib;
using namespace copylib_testing;
namespace ref = copylib_testing::reference_box;

namespace {

// a hand-built staged plan, so that this layer does not depend on apply_staging: gather on d0, cross, scatter on d1
copy_plan staged_plan(const copy_spec& spec) {
	const auto length = spec.source_layout.window_length();
	const auto source_buffer = staging_layout_from_fields(staging_id_from_fields(false, spec.source_device, 100), length);
	const auto target_buffer = staging_layout_from_fields(staging_id_from_fields(false, spec.target_device, 101), length);
	return {
	    spec_from_fields(spec.source_device, spec.source_layout, spec.source_device, source_buffer),
	    spec_from_fields(spec.source_device, source_buffer, spec.target_device, target_buffer),
	    spec_from_fields(spec.target_device, target_buffer, spec.target_device, spec.target_layout),
	};
}

copy_plan chunk(const copy_spec& spec, int64_t start, int64_t end) {
	return {spec_from_fields(
	    spec.source_device, with_window_fields(spec.source_layout, start, end), spec.target_device, with_window_fields(spec.target_layout, start, end))};
}

} // namespace

TEST_CASE("direct leaves every plan unchanged", "[d2d]") {
	const auto spec = ref::spec();
	for(const auto& plan : {copy_plan{spec}, staged_plan(spec)}) {
		std::vector<staging_request> log;
		const auto result = apply_d2d_implementation(plan, d2d_implementation::direct, recording_provider(log));
		CHECK(same_plan(result, plan));
		CHECK(log.empty());
	}
}

TEST_CASE("steps that are not between two devices are left alone", "[d2d]") {
	const auto d2d = GENERATE(d2d_implementation::direct, d2d_implementation::host_staging_at_source, d2d_implementation::host_staging_at_target,
	    d2d_implementation::host_staging_at_both);
	CAPTURE(d2d);

	for(const auto& [src, tgt] : {std::pair{device_id::d0, device_id::d0}, std::pair{device_id::host, device_id::d1}, std::pair{device_id::d0, device_id::host},
	        std::pair{device_id::host, device_id::host}}) {
		CAPTURE(src, tgt);
		std::vector<staging_request> log;
		const copy_plan plan{spec_from_fields(src, ref::fields(), tgt, ref::fields(0x80000))};
		const auto result = apply_d2d_implementation(plan, d2d, recording_provider(log));
		CHECK(same_plan(result, plan));
		CHECK(log.empty());
	}
}

TEST_CASE("host staging at one end splits a device-to-device copy in two", "[d2d]") {
	const auto d2d = GENERATE(d2d_implementation::host_staging_at_source, d2d_implementation::host_staging_at_target);
	CAPTURE(d2d);
	const auto spec = ref::spec(); // d0 -> d1, strided on both ends
	std::vector<staging_request> log;

	const auto plan = apply_d2d_implementation(copy_plan{spec}, d2d, recording_provider(log));

	CHECK(implements(plan, spec));
	REQUIRE(plan.size() == 2);
	CHECK(plan.at(0).source_device == device_id::d0);
	CHECK(same_fields(plan.at(0).source_layout, spec.source_layout));
	CHECK(plan.at(0).target_device == device_id::host);
	CHECK(is_staging_buffer(plan.at(0).target_layout, ref::total_bytes));
	CHECK(steps_connect(plan));
	CHECK(plan.at(1).target_device == device_id::d1);
	CHECK(same_fields(plan.at(1).target_layout, spec.target_layout));

	// the host buffer sits near the device the implementation names
	REQUIRE(log.size() == 1);
	CHECK(log.at(0).did == (d2d == d2d_implementation::host_staging_at_source ? device_id::d0 : device_id::d1));
	CHECK(log.at(0).on_host);
	CHECK(log.at(0).size == ref::total_bytes);
	CHECK(staging_is_consistent(plan));
}

TEST_CASE("host staging at both ends adds exactly one host-to-host copy", "[d2d]") {
	const auto spec = ref::spec();
	std::vector<staging_request> log;

	const auto plan = apply_d2d_implementation(copy_plan{spec}, d2d_implementation::host_staging_at_both, recording_provider(log));

	CHECK(implements(plan, spec));
	REQUIRE(plan.size() == 3);
	CHECK(plan.at(0).source_device == device_id::d0);
	CHECK(plan.at(0).target_device == device_id::host);
	CHECK(plan.at(1).source_device == device_id::host);
	CHECK(plan.at(1).target_device == device_id::host);
	CHECK(plan.at(2).source_device == device_id::host);
	CHECK(plan.at(2).target_device == device_id::d1);
	CHECK(is_staging_buffer(plan.at(1).source_layout, ref::total_bytes));
	CHECK(is_staging_buffer(plan.at(1).target_layout, ref::total_bytes));
	CHECK(space_of(plan.at(1).source_layout) != space_of(plan.at(1).target_layout));
	CHECK(steps_connect(plan));

	REQUIRE(log.size() == 2);
	CHECK(log.at(0).did == device_id::d0);
	CHECK(log.at(1).did == device_id::d1);
	CHECK(log.at(0).on_host);
	CHECK(log.at(1).on_host);
	CHECK(staging_is_consistent(plan));
}

TEST_CASE("staged plans only have their crossing rerouted", "[d2d]") {
	const auto spec = ref::spec();
	const auto plan = staged_plan(spec);
	const auto d2d = GENERATE(d2d_implementation::host_staging_at_source, d2d_implementation::host_staging_at_target, d2d_implementation::host_staging_at_both);
	CAPTURE(d2d);
	std::vector<staging_request> log;

	const auto result = apply_d2d_implementation(plan, d2d, recording_provider(log));

	CHECK(implements(result, spec));
	REQUIRE(result.size() == (d2d == d2d_implementation::host_staging_at_both ? 5u : 4u));
	CHECK(same_spec(result.front(), plan.front()));
	CHECK(same_spec(result.back(), plan.back()));
	CHECK(steps_connect(result));
	CHECK(staging_is_consistent(result));
}

TEST_CASE("after host staging no step goes directly between two devices", "[d2d]") {
	const auto d2d = GENERATE(d2d_implementation::host_staging_at_source, d2d_implementation::host_staging_at_target, d2d_implementation::host_staging_at_both);
	CAPTURE(d2d);
	const auto spec = ref::spec();

	for(const auto& plan : {copy_plan{spec}, staged_plan(spec)}) {
		std::vector<staging_request> log;
		const auto result = apply_d2d_implementation(plan, d2d, recording_provider(log));
		REQUIRE_FALSE(result.empty());
		for(const auto& step : result) {
			CHECK_FALSE(is_device_to_device(step));
		}
	}
}

TEST_CASE("rerouted steps keep their properties", "[d2d]") {
	const auto d2d = GENERATE(d2d_implementation::direct, d2d_implementation::host_staging_at_source, d2d_implementation::host_staging_at_target,
	    d2d_implementation::host_staging_at_both);
	const auto properties = GENERATE(copy_properties::none, copy_properties::use_kernel);
	CAPTURE(d2d, properties);
	std::vector<staging_request> log;

	const auto result = apply_d2d_implementation(copy_plan{ref::spec().with_properties(properties)}, d2d, recording_provider(log));

	REQUIRE_FALSE(result.empty());
	CHECK(all_steps_have(result, properties));
}

TEST_CASE("a chunk's host buffer holds only its window", "[d2d][window]") {
	const auto spec = ref::spec();
	std::vector<staging_request> log;

	const auto plan = apply_d2d_implementation(chunk(spec, 96, 192), d2d_implementation::host_staging_at_source, recording_provider(log));

	REQUIRE(plan.size() == 2);
	CHECK(is_staging_buffer(plan.at(0).target_layout, 96));
	REQUIRE(log.size() == 1);
	CHECK(log.at(0).size == 96);
}

TEST_CASE("the set overload reroutes every plan with its own buffers", "[d2d]") {
	const auto spec = ref::spec();
	const parallel_copy_set set{chunk(spec, 0, 96), chunk(spec, 96, 192), chunk(spec, 192, 288)};
	std::vector<staging_request> log;

	const auto result = apply_d2d_implementation(set, d2d_implementation::host_staging_at_both, recording_provider(log));

	CHECK(implements(result, spec));
	REQUIRE(result.size() == 3);
	CHECK(log.size() == 6);
	CHECK(staging_is_consistent(result));
}

TEST_CASE("the d2d implementation rejects an invalid plan", "[d2d][error]") {
	auto invalid = ref::spec();
	invalid.target_layout.end -= 8; // window lengths differ
	REQUIRE_FALSE(is_valid(invalid));
	std::vector<staging_request> log;

	CHECK_THROWS_AS(apply_d2d_implementation(copy_plan{invalid}, d2d_implementation::host_staging_at_source, recording_provider(log)), copylib::error);
	CHECK_NOTHROW(apply_d2d_implementation(copy_plan{ref::spec()}, d2d_implementation::host_staging_at_source, recording_provider(log)));
}
