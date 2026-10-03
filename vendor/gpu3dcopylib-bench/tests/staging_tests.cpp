#include "test_utils.hpp"

#include <copylib/core.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <utility>
#include <vector>

// Layer 5: apply_staging() turns one spec into a sequential plan [gather, middle, scatter], or a subset of it, so that
// the copy crossing devices is a single contiguous run.
//
// Pinned here: a side is staged exactly when its window is not one contiguous run; host-to-host copies are never
// staged, since they cross no bus; the strategy's properties replace the spec's on every step; staging buffers are 1D,
// sized to the window, with the window [0, length). A buffer lives on its side's device, and for a host side in host
// memory near the other device, as in the 2D library.

using namespace copylib;
using namespace copylib_testing;
namespace ref = copylib_testing::reference_box;

namespace {

copy_strategy staged(copy_properties properties = copy_properties::none) {
	return strategy_from_fields(copy_type::staged, properties, d2d_implementation::direct, 0);
}

// 288 contiguous bytes, the length of the reference box
data_layout contiguous(intptr_t at) { return normal_form::one_run(at, 0, ref::total_bytes); }

device_id staging_device_for(device_id side, device_id other) { return side == device_id::host ? other : side; }

using device_pair = std::pair<device_id, device_id>;

} // namespace

TEST_CASE("a direct strategy leaves the spec as it is", "[staging]") {
	std::vector<staging_request> log;
	const auto spec = ref::spec();

	const auto plan =
	    apply_staging(spec, strategy_from_fields(copy_type::direct, copy_properties::none, d2d_implementation::direct, 0), recording_provider(log));

	REQUIRE(plan.size() == 1);
	CHECK(same_spec(plan.front(), spec));
	CHECK(log.empty());
}

TEST_CASE("nothing is staged when both windows are contiguous", "[staging]") {
	const auto [src, tgt] =
	    GENERATE(device_pair{device_id::d0, device_id::d1}, device_pair{device_id::host, device_id::d1}, device_pair{device_id::d0, device_id::host});
	CAPTURE(src, tgt);

	SECTION("contiguous boxes") {
		std::vector<staging_request> log;
		const auto spec = spec_from_fields(src, contiguous(0x10000), tgt, contiguous(0x80000));
		const auto plan = apply_staging(spec, staged(), recording_provider(log));
		REQUIRE(plan.size() == 1);
		CHECK(same_spec(plan.front(), spec));
		CHECK(log.empty());
	}

	SECTION("windows inside a single row of a strided box") {
		// contiguity is judged on the window, so a strided box whose window stays within one row is not staged
		std::vector<staging_request> log;
		const auto spec = spec_from_fields(src, with_window_fields(ref::fields(), 2, 10), tgt, with_window_fields(ref::fields(0x80000), 26, 34));
		const auto plan = apply_staging(spec, staged(), recording_provider(log));
		REQUIRE(plan.size() == 1);
		CHECK(same_spec(plan.front(), spec));
		CHECK(log.empty());
	}
}

TEST_CASE("host-to-host copies are never staged", "[staging]") {
	// both sides strided, so everything but the host rule would stage them
	std::vector<staging_request> log;
	const auto spec = spec_from_fields(device_id::host, ref::fields(), device_id::host, shapes::six_rows_of_48(0x80000));

	const auto plan = apply_staging(spec, staged(), recording_provider(log));

	REQUIRE(plan.size() == 1);
	CHECK(same_spec(plan.front(), spec));
	CHECK(log.empty());
}

TEST_CASE("a strided source is gathered on its own device", "[staging]") {
	const auto [src, tgt] =
	    GENERATE(device_pair{device_id::d0, device_id::d1}, device_pair{device_id::d0, device_id::host}, device_pair{device_id::host, device_id::d1});
	CAPTURE(src, tgt);
	std::vector<staging_request> log;
	const auto spec = spec_from_fields(src, ref::fields(), tgt, contiguous(0x80000));

	const auto plan = apply_staging(spec, staged(), recording_provider(log));

	CHECK(implements(plan, spec));
	REQUIRE(plan.size() == 2);
	const auto& gather = plan.at(0);
	const auto& middle = plan.at(1);

	CHECK(gather.source_device == src);
	CHECK(same_fields(gather.source_layout, spec.source_layout));
	CHECK(gather.target_device == src);
	CHECK(is_staging_buffer(gather.target_layout, ref::total_bytes));

	CHECK(steps_connect(plan));
	CHECK(middle.target_device == tgt);
	CHECK(same_fields(middle.target_layout, spec.target_layout));

	REQUIRE(log.size() == 1);
	CHECK(log.at(0).did == staging_device_for(src, tgt));
	CHECK(log.at(0).on_host == (src == device_id::host));
	CHECK(log.at(0).size == ref::total_bytes);
	CHECK(staging_is_consistent(plan));
}

TEST_CASE("a strided target is scattered on its own device", "[staging]") {
	const auto [src, tgt] =
	    GENERATE(device_pair{device_id::d0, device_id::d1}, device_pair{device_id::host, device_id::d1}, device_pair{device_id::d0, device_id::host});
	CAPTURE(src, tgt);
	std::vector<staging_request> log;
	const auto spec = spec_from_fields(src, contiguous(0x10000), tgt, ref::fields(0x80000));

	const auto plan = apply_staging(spec, staged(), recording_provider(log));

	CHECK(implements(plan, spec));
	REQUIRE(plan.size() == 2);
	const auto& middle = plan.at(0);
	const auto& scatter = plan.at(1);

	CHECK(middle.source_device == src);
	CHECK(same_fields(middle.source_layout, spec.source_layout));
	CHECK(steps_connect(plan));

	CHECK(scatter.source_device == tgt);
	CHECK(is_staging_buffer(scatter.source_layout, ref::total_bytes));
	CHECK(scatter.target_device == tgt);
	CHECK(same_fields(scatter.target_layout, spec.target_layout));

	REQUIRE(log.size() == 1);
	CHECK(log.at(0).did == staging_device_for(tgt, src));
	CHECK(log.at(0).on_host == (tgt == device_id::host));
	CHECK(log.at(0).size == ref::total_bytes);
	CHECK(staging_is_consistent(plan));
}

TEST_CASE("strided on both ends is gathered then crossed once contiguously then scattered", "[staging]") {
	const auto [src, tgt] =
	    GENERATE(device_pair{device_id::d0, device_id::d1}, device_pair{device_id::host, device_id::d1}, device_pair{device_id::d0, device_id::host});
	CAPTURE(src, tgt);
	std::vector<staging_request> log;
	const auto spec = spec_from_fields(src, ref::fields(), tgt, shapes::six_rows_of_48(0x80000));

	const auto plan = apply_staging(spec, staged(), recording_provider(log));

	CHECK(implements(plan, spec));
	REQUIRE(plan.size() == 3);
	const auto& gather = plan.at(0);
	const auto& middle = plan.at(1);
	const auto& scatter = plan.at(2);

	// only the middle copy crosses devices, and it is one contiguous run on both ends
	CHECK_FALSE(crosses_devices(gather));
	CHECK_FALSE(crosses_devices(scatter));
	CHECK(middle.source_device == src);
	CHECK(middle.target_device == tgt);
	CHECK(is_one_run(middle.source_layout));
	CHECK(is_one_run(middle.target_layout));

	CHECK(is_staging_buffer(gather.target_layout, ref::total_bytes));
	CHECK(is_staging_buffer(scatter.source_layout, ref::total_bytes));
	CHECK(space_of(gather.target_layout) != space_of(scatter.source_layout));
	CHECK(steps_connect(plan));
	CHECK(staging_is_consistent(plan));

	REQUIRE(log.size() == 2);
	CHECK(is_valid(plan));
}

TEST_CASE("the strategy's properties replace the spec's", "[staging]") {
	// on every path through apply_staging, including the ones that return the spec unstaged
	const auto path = GENERATE(0, 1, 2, 3);
	const bool strategy_sets_them = GENERATE(true, false);
	CAPTURE(path, strategy_sets_them);

	const auto spec_properties = strategy_sets_them ? copy_properties::none : copy_properties::use_kernel;
	const auto strategy_properties = strategy_sets_them ? copy_properties::use_kernel : copy_properties::none;
	const auto type = path == 0 ? copy_type::direct : copy_type::staged;

	copy_spec spec = ref::spec(); // path 0 and 3: strided on both ends
	if(path == 1) { spec = spec_from_fields(device_id::d0, contiguous(0x10000), device_id::d1, contiguous(0x80000)); }
	if(path == 2) { spec = spec_from_fields(device_id::host, ref::fields(), device_id::host, ref::fields(0x80000)); }
	spec.properties = spec_properties;

	std::vector<staging_request> log;
	const auto plan = apply_staging(spec, strategy_from_fields(type, strategy_properties, d2d_implementation::direct, 0), recording_provider(log));

	REQUIRE_FALSE(plan.empty());
	CHECK(all_steps_have(plan, strategy_properties));
}

TEST_CASE("a staged chunk gets a buffer holding exactly its window from offset 0", "[staging][window]") {
	const auto [start, end] = GENERATE(std::pair<int64_t, int64_t>{96, 192}, std::pair<int64_t, int64_t>{3, 50});
	CAPTURE(start, end);
	const auto spec = spec_from_fields(
	    device_id::d0, with_window_fields(ref::fields(), start, end), device_id::d1, with_window_fields(shapes::six_rows_of_48(0x80000), start, end));
	std::vector<staging_request> log;

	const auto plan = apply_staging(spec, staged(), recording_provider(log));

	CHECK(implements(plan, spec));
	REQUIRE(plan.size() == 3);
	CHECK(is_staging_buffer(plan.at(0).target_layout, end - start));
	CHECK(is_staging_buffer(plan.at(2).source_layout, end - start));
	REQUIRE(log.size() == 2);
	CHECK(log.at(0).size == end - start);
	CHECK(log.at(1).size == end - start);
}

TEST_CASE("a reshaping spec is staged on both ends", "[staging]") {
	// 12 rows of 24 bytes into 6 rows of 48
	std::vector<staging_request> log;
	const auto spec = spec_from_fields(device_id::d0, ref::fields(), device_id::d1, shapes::six_rows_of_48(0x80000));

	const auto plan = apply_staging(spec, staged(), recording_provider(log));

	CHECK(implements(plan, spec));
	CHECK(plan.size() == 3);
	CHECK(staging_is_consistent(plan));
}

TEST_CASE("a same-device copy is staged correctly", "[staging]") {
	// Only correctness is pinned. Whether staging within one device is worth doing at all is still open, so the
	// shape of this plan is deliberately left free.
	std::vector<staging_request> log;
	const auto spec = spec_from_fields(device_id::d0, ref::fields(), device_id::d0, shapes::six_rows_of_48(0x80000));

	const auto plan = apply_staging(spec, staged(), recording_provider(log));

	CHECK(implements(plan, spec));
	CHECK(steps_connect(plan));
	CHECK(staging_is_consistent(plan));
	CHECK(is_valid(plan));
}

TEST_CASE("the set overload stages every plan with its own buffers", "[staging]") {
	const auto spec = ref::spec();
	const auto chunk = [&](int64_t start, int64_t end) {
		return copy_plan{spec_from_fields(
		    device_id::d0, with_window_fields(spec.source_layout, start, end), device_id::d1, with_window_fields(spec.target_layout, start, end))};
	};
	const parallel_copy_set set{chunk(0, 96), chunk(96, 192), chunk(192, 288)};
	std::vector<staging_request> log;

	const auto staged_set = apply_staging(set, staged(), recording_provider(log));

	// the oracle also rejects plans of a set that share a staging buffer
	CHECK(implements(staged_set, spec));
	REQUIRE(staged_set.size() == 3);
	for(const auto& plan : staged_set) {
		CHECK(plan.size() == 3);
	}
	CHECK(log.size() == 6);
	CHECK(staging_is_consistent(staged_set));
}

TEST_CASE("the basic staging provider hands out a distinct id per request", "[staging][provider]") {
	SECTION("called directly") {
		basic_staging_provider provider;
		const auto a = provider(device_id::d0, false, 128);
		const auto b = provider(device_id::d1, true, 256);
		const auto c = provider(device_id::d0, false, 128);

		CHECK(a != b);
		CHECK(a != c);
		CHECK(b != c);
		CHECK(a.is_staging_id == staging_id::staging_id_flag);
		CHECK(b.did == device_id::d1);
		CHECK(b.on_host);
		CHECK_FALSE(c.on_host);
	}

	SECTION("through a staging_buffer_provider") {
		// std::function holds its own copy of the provider, which keeps counting across calls
		const staging_buffer_provider provider = basic_staging_provider{};
		CHECK(provider(device_id::d0, false, 64) != provider(device_id::d0, false, 64));
	}
}

TEST_CASE("staging rejects an invalid spec", "[staging][error]") {
	auto invalid = ref::spec();
	invalid.target_layout.end -= 8; // window lengths differ
	REQUIRE_FALSE(is_valid(invalid));
	std::vector<staging_request> log;

	CHECK_THROWS_AS(apply_staging(invalid, staged(), recording_provider(log)), copylib::error);
	CHECK_NOTHROW(apply_staging(ref::spec(), staged(), recording_provider(log)));
}
