cmake_minimum_required(VERSION 3.24)

project(gpu3dcopylib CXX)

option(COPYLIB_USE_FMT "Use the fmt library rather than std::format" OFF)
# only built by default for top-level builds, so that vendoring this library does not pull in Catch2
option(COPYLIB_BUILD_TESTS "Build the test suite" ${PROJECT_IS_TOP_LEVEL})

# when vendored, the enclosing project has usually already found a SYCL implementation
if(NOT COMMAND add_sycl_to_target)
  find_package(SimSYCL CONFIG QUIET)
  if(SimSYCL_FOUND)
    message(STATUS "copylib: using SimSYCL ${SimSYCL_VERSION} (${SimSYCL_DIR})")
  else()
    find_package(AdaptiveCpp CONFIG QUIET)
    if(AdaptiveCpp_FOUND)
      message(STATUS "copylib: using AdaptiveCpp ${AdaptiveCpp_VERSION} (${AdaptiveCpp_DIR})")
    else()
      find_package(IntelSYCL CONFIG QUIET)
      if(IntelSYCL_FOUND)
        message(STATUS "copylib: using IntelSYCL ${IntelSYCL_VERSION} (${IntelSYCL_DIR})")
      else()
        message(FATAL_ERROR "copylib: no SYCL implementation (SimSYCL, AdaptiveCpp or IntelSYCL) found in CMAKE_PREFIX_PATH")
      endif()
    endif()
  endif()
endif()

set(COPYLIB_SOURCES
  src/core.cpp
  src/backend.cpp
  src/backend_kernels.cpp
  src/support.cpp
  src/utils.cpp
)

add_library(copylib ${COPYLIB_SOURCES})
add_library(copylib::copylib ALIAS copylib)

# Only the backend actually needs SYCL compilation, the others just need the libs/include path, but hard to make work consistently over different SYCL implementations
add_sycl_to_target(TARGET copylib SOURCES ${COPYLIB_SOURCES})

target_include_directories(copylib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
# BS::thread_pool 5.0.0 (MIT, https://github.com/bshoshany/thread-pool), vendored as in the 2D library. Public, since the
# executor owns a pool and BS::thread_pool is a class template that cannot usefully be forward-declared.
target_include_directories(copylib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/vendor>)
target_compile_features(copylib PUBLIC cxx_std_20)

# the public headers expose SYCL types, but SimSYCL only links itself privately in add_sycl_to_target
if(TARGET SimSYCL::simsycl)
  target_link_libraries(copylib PUBLIC SimSYCL::simsycl)
endif()

if(AdaptiveCpp_FOUND)
  target_compile_definitions(copylib PUBLIC ACPP_ALLOW_INSTANT_SUBMISSION=1) # massive performance improvement for fine-grained kernels
endif()

if(COPYLIB_USE_FMT)
  find_package(fmt REQUIRED)
  target_link_libraries(copylib PUBLIC fmt::fmt)
  target_compile_definitions(copylib PUBLIC COPYLIB_USE_FMT)
endif()

if(COPYLIB_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
