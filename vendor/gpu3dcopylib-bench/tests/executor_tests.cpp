#include "backend_test_utils.hpp"

#include <copylib/backend.hpp>
#include <copylib/utils.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <pthread.h>

// Layer 12: constructing the executor and what it reports about itself.
//
// The executor creates and owns one in-order queue per device and queue index, on the devices and in the contexts it is
// given, and staging memory of the given size. It allocates no data buffers and does not configure the SYCL system; the
// harness does both (backend_test_utils.hpp).

using namespace copylib;
using namespace copylib_testing;
namespace ref = copylib_testing::reference_box;

using device_contexts = std::vector<std::pair<sycl::device, sycl::context>>;

namespace {

// sets an environment variable for its lifetime
struct scoped_env {
	const char* name;
	scoped_env(const char* name, const std::string& value) : name(name) { setenv(name, value.c_str(), 1); }
	scoped_env(const scoped_env&) = delete;
	scoped_env& operator=(const scoped_env&) = delete;
	~scoped_env() { unsetenv(name); }
};

cpu_set_t current_affinity() {
	cpu_set_t mask;
	CPU_ZERO(&mask);
	REQUIRE(pthread_getaffinity_np(pthread_self(), sizeof(mask), &mask) == 0);
	return mask;
}

int first_allowed_cpu() {
	const auto mask = current_affinity();
	for(int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
		if(CPU_ISSET(cpu, &mask)) { return cpu; }
	}
	return -1;
}

} // namespace

TEST_CASE("an executor has in-order queues for every requested device and queue index", "[executor]") {
	const auto devices = test_devices(2);
	executor exec(test_staging_bytes, devices, 3);

	CHECK(exec.get_queues_per_device() == 3);
	for(size_t i = 0; i < devices.size(); ++i) {
		const auto did = static_cast<device_id>(i);
		for(int64_t q = 0; q < 3; ++q) {
			CAPTURE(did, q);
			CHECK(exec.get_queue(did, q).is_in_order());
			CHECK(exec.get_queue(did, q).get_device() == devices[i]);
		}
	}
	CHECK(exec.get_queue(device_id::d0, 0) != exec.get_queue(device_id::d0, 1));
}

TEST_CASE("an executor creates its queues in the context given with each device", "[executor]") {
	// one context per device, as Celerity creates them
	const auto devices = test_devices(2);
	const device_contexts contexts{{devices[0], sycl::context(devices[0])}, {devices[1], sycl::context(devices[1])}};
	executor exec(test_staging_bytes, contexts, 2);

	for(size_t i = 0; i < contexts.size(); ++i) {
		const auto did = static_cast<device_id>(i);
		for(int64_t q = 0; q < 2; ++q) {
			CAPTURE(did, q);
			CHECK(exec.get_queue(did, q).get_device() == contexts[i].first);
			CHECK(exec.get_queue(did, q).get_context() == contexts[i].second);
		}
	}

	// a staged copy within one device, on memory and staging buffers of that device's own context
	const prepared_copy copy(exec, on_device(device_id::d1), ref::fields(), on_device(device_id::d1), shapes::six_rows_of_48(0));
	const auto outcome = run_copy(exec, copy, strategy_from_fields(copy_type::staged, copy_properties::use_kernel, d2d_implementation::direct, 64));
	INFO(outcome.describe());
	CHECK(outcome.correct());
}

TEST_CASE("an executor rejects a device paired with a context that does not contain it", "[executor][error]") {
	configure_test_system();
	const auto all = sycl::device::get_devices();
	if(all.size() < 2) { SKIP("needs two distinct devices"); }

	CHECK_THROWS_AS(executor(test_staging_bytes, device_contexts{{all[0], sycl::context(all[1])}}, 1), copylib::error);
	CHECK_NOTHROW(executor(test_staging_bytes, device_contexts{{all[0], sycl::context(all[0])}}, 1));
}

TEST_CASE("the executor's buffer size is its staging memory per device", "[executor]") {
	// rounded up to a whole number of the 128-byte staging alignment
	const auto exec = make_executor(1, 1, 12345 * 64);
	CHECK(exec.get_buffer_size() == 12346 * 64);
}

TEST_CASE("the staging memory is divided evenly among the workers", "[executor][staging]") {
	// one worker per queue index; 790144 / 2 = 395072 bytes, rounded down to the 128-byte staging alignment
	const auto exec = make_executor(1, 2, 12345 * 64);
	CHECK(exec.get_staging_slice_size() == 3086 * 128);
}

TEST_CASE("an executor needs staging memory for every worker", "[executor][staging][error]") {
	// none, negative, and 128 bytes, which leave each of two workers less than the 128-byte staging alignment
	CHECK_THROWS_AS(executor(0, test_devices(1), 1), copylib::error);
	CHECK_THROWS_AS(executor(-128, test_devices(1), 1), copylib::error);
	CHECK_THROWS_AS(executor(128, test_devices(1), 2), copylib::error);
	CHECK_NOTHROW(executor(256, test_devices(1), 2));
}

// a buffer size alone does not convert into an executor
static_assert(!std::is_convertible_v<int64_t, executor>);

TEST_CASE("an executor cannot have more devices than device_id can name", "[executor][error]") {
	CHECK_THROWS_AS(executor(test_staging_bytes, test_devices(9), 1), copylib::error);
	CHECK_NOTHROW(executor(test_staging_bytes, test_devices(8), 1));
}

TEST_CASE("the GPU shortcut cannot pick more GPUs than exist", "[executor][error]") {
	configure_test_system();
	// more than any machine running the tests
	CHECK_THROWS_AS(executor(test_staging_bytes, 64, 1), copylib::error);
}

TEST_CASE("an executor needs at least one device and one queue per device", "[executor][error]") {
	CHECK_THROWS_AS(executor(test_staging_bytes, std::vector<sycl::device>{}, 1), copylib::error);
	CHECK_THROWS_AS(executor(test_staging_bytes, test_devices(1), 0), copylib::error);
	CHECK_THROWS_AS(executor(test_staging_bytes, 0, 1), copylib::error);
	CHECK_NOTHROW(executor(test_staging_bytes, test_devices(1), 1));
}

TEST_CASE("an executor describes itself", "[executor]") {
	const auto exec = make_executor();
	CHECK_FALSE(exec.get_sycl_impl_name().empty());
	CHECK_FALSE(exec.get_info().empty());
}

TEST_CASE("host staging is pinned only when COPYLIB_ALLOC_CPU_IDS asks for it", "[executor][pinning]") {
	unsetenv("COPYLIB_ALLOC_CPU_IDS");
	CHECK_FALSE(get_cpu_for_gpu_alloc(0, 2).has_value());
	CHECK(make_executor().get_info().find("host alloc not pinned") != std::string::npos);

	// a CPU this thread may run on, for both devices; afterwards the thread runs where it ran before
	const int cpu = first_allowed_cpu();
	REQUIRE(cpu >= 0);
	const auto before = current_affinity();
	{
		const scoped_env ids("COPYLIB_ALLOC_CPU_IDS", std::to_string(cpu) + "," + std::to_string(cpu));
		CHECK(get_cpu_for_gpu_alloc(1, 2) == cpu);
		CHECK(make_executor().get_info().find("host alloc on core " + std::to_string(cpu)) != std::string::npos);
	}
	const auto after = current_affinity();
	CHECK(CPU_EQUAL(&before, &after));
}

TEST_CASE("an invalid COPYLIB_ALLOC_CPU_IDS is reported as copylib::error", "[executor][pinning][error]") {
	// malformed, too few for two devices, negative, and a CPU this machine almost certainly lacks
	const auto before = current_affinity();
	for(const char* ids : {"0,x", "0, 1", "0", "-1,0", "1023,1023"}) {
		CAPTURE(ids);
		const scoped_env set("COPYLIB_ALLOC_CPU_IDS", ids);
		CHECK_THROWS_AS(make_executor(2, 1), copylib::error);
	}
	const auto after = current_affinity();
	CHECK(CPU_EQUAL(&before, &after));
}

TEST_CASE("COPYLIB_WG_SIZE overrides the work-group size of the copy kernels", "[executor]") {
	unsetenv("COPYLIB_WG_SIZE");
	CHECK(make_executor().get_preferred_wg_size() > 0);

	const scoped_env wg_size("COPYLIB_WG_SIZE", "48");
	CHECK(make_executor().get_preferred_wg_size() == 48);
}

TEST_CASE("an invalid COPYLIB_WG_SIZE is reported as copylib::error", "[executor][error]") {
	// empty, malformed, zero, which would divide by zero when sizing the kernels, negative, and beyond int32_t
	for(const char* size : {"", "x", "64x", " 64", "0", "-32", "4294967296"}) {
		CAPTURE(size);
		const scoped_env set("COPYLIB_WG_SIZE", size);
		CHECK_THROWS_AS(make_executor(), copylib::error);
	}
}

TEST_CASE("failed checks throw copylib::error", "[executor][error]") {
	// COPYLIB_ENSURE is the library's single way of failing; it throws rather than ending the process
	CHECK_THROWS_AS([] { COPYLIB_ENSURE(1 + 1 == 3, "arithmetic is broken: {}", 1 + 1); }(), copylib::error);
	CHECK_NOTHROW([] { COPYLIB_ENSURE(1 + 1 == 2, "arithmetic is broken: {}", 1 + 1); }());
}
