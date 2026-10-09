#include "test_utils.hpp"

#include <copylib/core.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
#include <vector>

// Layer 7: manifest_strategy() = normalize -> apply_chunking -> apply_staging -> apply_d2d_implementation.
//
// Every combination of copy type, properties, d2d implementation and chunk size runs on a set of named specs, then on
// randomly generated ones, and each result is checked against the oracle and the pipeline's invariants. Failures are
// counted and only the first per invariant is reported, so a broken pipeline cannot flood the CI log. Specs and
// strategies are described by hand rather than through the formatters, so the reports do not depend on layer 8.

using namespace copylib;
using namespace copylib_testing;
namespace ref = copylib_testing::reference_box;

namespace {

std::string describe(device_id did) { return did == device_id::host ? "host" : "d" + std::to_string(static_cast<int>(did)); }

std::string describe(const data_layout& l) {
	const auto s = [](int64_t v) { return std::to_string(v); };
	return "[" + s(l.d0_stride) + "x" + s(l.d1_stride) + " d0:[" + s(l.d0_start_offset) + "," + s(l.d0_end_offset) + ") d1:[" + s(l.d1_start_offset) + ","
	       + s(l.d1_end_offset) + ") d2:[" + s(l.d2_start_offset) + "," + s(l.d2_end_offset) + ") win:[" + s(l.start) + "," + s(l.end) + ")]";
}

std::string describe(const copy_spec& spec) {
	return describe(spec.source_device) + describe(spec.source_layout) + " -> " + describe(spec.target_device) + describe(spec.target_layout);
}

std::string describe(const copy_strategy& s) {
	const char* d2d_names[] = {"direct", "host_staging_at_source", "host_staging_at_target", "host_staging_at_both"};
	return std::string(s.type == copy_type::staged ? "staged" : "direct") + ", " + ((s.properties & copy_properties::use_kernel) ? "use_kernel" : "none")
	       + ", d2d " + d2d_names[static_cast<int>(s.d2d)] + ", chunk " + std::to_string(s.chunk_size);
}

// What every manifested set must satisfy, whatever the spec and strategy.
const std::vector<std::string> invariant_names = {
    "implements the spec",
    "is a valid, non-empty set",
    "every step carries the strategy's properties",
    "the steps of each plan connect",
    "staging buffers are used consistently",
    "a staged copy crosses devices only in contiguous runs",
    "with host staging no step goes directly between two devices",
    "no chunk is larger than chunk_size allows",
};

struct invariant_report {
	int combinations = 0;
	std::map<std::string, int> failures;
	std::map<std::string, std::string> first_failure;

	void expect(bool holds, const std::string& invariant, const std::string& where) {
		if(holds) return;
		if(failures[invariant]++ == 0) first_failure[invariant] = where;
	}

	void check(const copy_spec& spec, const copy_strategy& strategy, const parallel_copy_set& set, const std::string& where) {
		++combinations;
		expect(implements(set, spec), invariant_names[0], where);
		expect(!set.empty() && is_valid(set), invariant_names[1], where);
		expect(all_steps_have(set, strategy.properties), invariant_names[2], where);

		bool connected = true;
		bool contiguous_crossings = true;
		bool no_direct_d2d = true;
		bool bounded = true;
		const int64_t largest_chunk = std::max(strategy.chunk_size, copy_alignment(spec));
		for(const auto& plan : set) {
			connected = connected && steps_connect(plan);
			for(const auto& step : plan) {
				if(strategy.type == copy_type::staged && crosses_devices(step)) {
					contiguous_crossings = contiguous_crossings && is_one_run(step.source_layout) && is_one_run(step.target_layout);
				}
				if(strategy.d2d != d2d_implementation::direct) { no_direct_d2d = no_direct_d2d && !is_device_to_device(step); }
				if(strategy.chunk_size > 0) { bounded = bounded && step.source_layout.window_length() <= largest_chunk; }
			}
		}
		expect(connected, invariant_names[3], where);
		expect(staging_is_consistent(set), invariant_names[4], where);
		expect(contiguous_crossings, invariant_names[5], where);
		expect(no_direct_d2d, invariant_names[6], where);
		expect(bounded, invariant_names[7], where);
	}

	void report() {
		for(const auto& name : invariant_names) {
			INFO(name << ": " << failures[name] << " of " << combinations << " combinations fail, the first being " << first_failure[name]);
			CHECK(failures[name] == 0);
		}
	}
};

struct named_spec {
	std::string name;
	copy_spec spec;
};

std::vector<named_spec> named_specs() {
	const auto shifted =
	    spec_from_fields(device_id::d0, with_window_fields(ref::fields(), 0, 280), device_id::d1, with_window_fields(ref::fields(0x80000), 8, 288));
	const auto partial =
	    spec_from_fields(device_id::d0, with_window_fields(ref::fields(), 3, 285), device_id::d2, with_window_fields(ref::fields(0x80000), 3, 285));
	return {
	    {"reference box d0 -> d1", ref::spec()},
	    {"reference box host -> d1", spec_from_fields(device_id::host, ref::fields(), device_id::d1, ref::fields(0x80000))},
	    {"reference box d0 -> host", spec_from_fields(device_id::d0, ref::fields(), device_id::host, ref::fields(0x80000))},
	    {"reference box d0 -> d0", spec_from_fields(device_id::d0, ref::fields(), device_id::d0, ref::fields(0x80000))},
	    {"reference box host -> host", spec_from_fields(device_id::host, ref::fields(), device_id::host, ref::fields(0x80000))},
	    {"strided -> contiguous", spec_from_fields(device_id::d0, ref::fields(), device_id::d1, normal_form::one_run(0x80000, 0, ref::total_bytes))},
	    {"contiguous -> strided", spec_from_fields(device_id::d1, normal_form::one_run(0x10000, 0, ref::total_bytes), device_id::d0, ref::fields(0x80000))},
	    {"reshaping 24-byte rows into 48-byte rows", spec_from_fields(device_id::d0, ref::fields(), device_id::d1, shapes::six_rows_of_48(0x80000))},
	    {"unaligned window [3, 285)", partial},
	    {"windows shifted by 8 bytes", shifted},
	};
}

std::vector<copy_strategy> all_strategies() {
	std::vector<copy_strategy> strategies;
	for(const auto type : {copy_type::direct, copy_type::staged}) {
		for(const auto properties : {copy_properties::none, copy_properties::use_kernel}) {
			for(const auto d2d : {d2d_implementation::direct, d2d_implementation::host_staging_at_source, d2d_implementation::host_staging_at_target,
			        d2d_implementation::host_staging_at_both}) {
				for(const int64_t chunk_size : {0, 1, 8, 64, 77, 1000}) {
					strategies.push_back(strategy_from_fields(type, properties, d2d, chunk_size));
				}
			}
		}
	}
	return strategies;
}

// Uniform in [lo, hi]. Plain modulo rather than std::uniform_int_distribution, whose output differs between standard
// libraries, so that a seed reproduces the same cases everywhere.
int64_t pick(std::mt19937_64& rng, int64_t lo, int64_t hi) { return lo + static_cast<int64_t>(rng() % static_cast<uint64_t>(hi - lo + 1)); }

// A random box in an allocation of up to 64-byte rows, 5 rows per plane and 6 planes, with its window over the whole box.
data_layout random_box(std::mt19937_64& rng, intptr_t base) {
	const int64_t elem = int64_t{1} << pick(rng, 0, 3);
	const int64_t row_elems = pick(rng, 1, 8);
	const int64_t d0_start = pick(rng, 0, row_elems - 1);
	const int64_t d0_end = pick(rng, d0_start + 1, row_elems);
	const int64_t d1_stride = pick(rng, 1, 5);
	const int64_t d1_start = pick(rng, 0, d1_stride - 1);
	const int64_t d1_end = pick(rng, d1_start + 1, d1_stride);
	const int64_t d2_start = pick(rng, 0, 3);
	const int64_t d2_end = d2_start + pick(rng, 1, 3);
	const int64_t total = (d0_end - d0_start) * elem * (d1_end - d1_start) * (d2_end - d2_start);
	return layout_from_fields(base, row_elems * elem, d1_stride, d0_start * elem, d1_start, d2_start, d0_end * elem, d1_end, d2_end, 0, total);
}

copy_spec random_spec(std::mt19937_64& rng) {
	constexpr device_id devices[] = {device_id::host, device_id::d0, device_id::d1, device_id::d2};

	auto source = random_box(rng, 0x100000);
	const int64_t source_total = source.end;
	source.start = pick(rng, 0, source_total - 1);
	source.end = pick(rng, source.start + 1, source_total);
	const int64_t length = source.end - source.start;

	// a random target shape large enough for the window, falling back to a contiguous run
	data_layout target = normal_form::one_run(0x800000, pick(rng, 0, 64), length);
	for(int attempt = 0; attempt < 20; ++attempt) {
		auto candidate = random_box(rng, 0x800000);
		if(candidate.end < length) continue;
		candidate.start = pick(rng, 0, candidate.end - length);
		candidate.end = candidate.start + length;
		target = candidate;
		break;
	}
	return spec_from_fields(devices[pick(rng, 0, 3)], source, devices[pick(rng, 0, 3)], target);
}

copy_strategy random_strategy(std::mt19937_64& rng) {
	constexpr int64_t chunk_sizes[] = {0, 1, 3, 8, 16, 40, 64, 1000};
	return strategy_from_fields(pick(rng, 0, 1) == 0 ? copy_type::direct : copy_type::staged,
	    pick(rng, 0, 1) == 0 ? copy_properties::none : copy_properties::use_kernel, static_cast<d2d_implementation>(pick(rng, 0, 3)),
	    chunk_sizes[pick(rng, 0, 7)]);
}

} // namespace

TEST_CASE("manifest_strategy chains normalization, chunking, staging and the d2d implementation", "[manifest]") {
	// design.md, "Pipeline recap"; two fresh providers hand out the same ids when called in the same order
	const auto spec = ref::spec();
	const auto strategy = strategy_from_fields(copy_type::staged, copy_properties::use_kernel, d2d_implementation::host_staging_at_both, 64);
	std::vector<staging_request> manifest_log;
	std::vector<staging_request> chain_log;
	const auto chain_provider = recording_provider(chain_log);

	const auto manifested = manifest_strategy(spec, strategy, recording_provider(manifest_log));
	const auto chained =
	    apply_d2d_implementation(apply_staging(apply_chunking(normalize(spec), strategy), strategy, chain_provider), strategy.d2d, chain_provider);

	// without this, two pipelines that both return an empty set would compare equal
	REQUIRE_FALSE(manifested.empty());
	CHECK(same_set(manifested, chained));
}

TEST_CASE("manifest_strategy normalizes the spec before chunking", "[manifest]") {
	// 192 adjacent rows of 80 bytes are one contiguous run, so a direct copy of them is planned as a single 1D run
	constexpr int64_t length = 12 * ref::plane_bytes;
	const auto spec = spec_from_fields(device_id::d0, shapes::whole_allocation(), device_id::d1, shapes::whole_allocation(0x80000));
	const auto strategy = strategy_from_fields(copy_type::direct, copy_properties::none, d2d_implementation::direct, 0);
	std::vector<staging_request> log;

	const auto set = manifest_strategy(spec, strategy, recording_provider(log));

	REQUIRE(set.size() == 1);
	REQUIRE(set.front().size() == 1);
	CHECK(same_fields(set.front().front().source_layout, normal_form::one_run(ref::base, 0, length)));
	CHECK(same_fields(set.front().front().target_layout, normal_form::one_run(0x80000, 0, length)));
}

TEST_CASE("every strategy on every named spec", "[manifest]") {
	invariant_report report;
	for(const auto& [name, spec] : named_specs()) {
		for(const auto& strategy : all_strategies()) {
			std::vector<staging_request> log;
			const auto set = manifest_strategy(spec, strategy, recording_provider(log));
			report.check(spec, strategy, set, name + " with " + describe(strategy));
		}
	}
	report.report();
}

TEST_CASE("randomized specs and strategies", "[manifest][random]") {
	// set COPYLIB_TEST_SEED to reproduce a reported case; cases are generated in sequence, so the index identifies it
	const char* seed_override = std::getenv("COPYLIB_TEST_SEED");
	const uint64_t seed = seed_override != nullptr ? std::strtoull(seed_override, nullptr, 10) : 20260922;
	CAPTURE(seed);
	std::mt19937_64 rng(seed);

	invariant_report report;
	for(int i = 0; i < 400; ++i) {
		const auto spec = random_spec(rng);
		const auto strategy = random_strategy(rng);
		std::vector<staging_request> log;
		const auto set = manifest_strategy(spec, strategy, recording_provider(log));
		report.check(spec, strategy, set, "case " + std::to_string(i) + ": " + describe(spec) + " with " + describe(strategy));
	}
	report.report();
}

TEST_CASE("manifest_strategy rejects an invalid spec", "[manifest][error]") {
	auto invalid = ref::spec();
	invalid.target_layout.end -= 8; // window lengths differ
	REQUIRE_FALSE(is_valid(invalid));
	const auto strategy = strategy_from_fields(copy_type::staged, copy_properties::none, d2d_implementation::host_staging_at_both, 64);
	std::vector<staging_request> log;

	CHECK_THROWS_AS(manifest_strategy(invalid, strategy, recording_provider(log)), copylib::error);
	CHECK_NOTHROW(manifest_strategy(ref::spec(), strategy, recording_provider(log)));
}
