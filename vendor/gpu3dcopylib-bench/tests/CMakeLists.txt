include(FetchContent)

# FIND_PACKAGE_ARGS lets an installed Catch2 (e.g. in the dev container) satisfy this without a download
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.7.1
  FIND_PACKAGE_ARGS NAMES Catch2
)
FetchContent_MakeAvailable(Catch2)

set(TEST_FILES
  utils_tests.cpp
  layout_tests.cpp
  validation_tests.cpp
  reference_tests.cpp
  normalization_tests.cpp
  chunking_tests.cpp
  staging_tests.cpp
  d2d_tests.cpp
  manifest_tests.cpp
  support_tests.cpp
  backend_tests.cpp
  execution_tests.cpp
  executor_tests.cpp
)

# Build one object library per test file so that each file can also be run as its own executable without
# compiling it twice. AdaptiveCpp trips over the per-file executables with a CUDA fat binary linker error,
# so it only gets the combined one.
if(AdaptiveCpp_FOUND)
  add_executable(all_tests ${TEST_FILES})
  add_sycl_to_target(TARGET all_tests SOURCES ${TEST_FILES})
  target_link_libraries(all_tests PRIVATE copylib::copylib Catch2::Catch2WithMain)
  target_compile_options(all_tests PRIVATE -Wall -Wextra)
else()
  foreach(TEST_FILE ${TEST_FILES})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_library(${TEST_NAME}_obj OBJECT ${TEST_FILE})
    add_sycl_to_target(TARGET ${TEST_NAME}_obj SOURCES ${TEST_FILE})
    target_link_libraries(${TEST_NAME}_obj PRIVATE copylib::copylib Catch2::Catch2)
    target_compile_options(${TEST_NAME}_obj PRIVATE -Wall -Wextra)
    add_executable(${TEST_NAME})
    target_link_libraries(${TEST_NAME} PRIVATE ${TEST_NAME}_obj Catch2::Catch2WithMain)
    list(APPEND TEST_OBJECTS ${TEST_NAME}_obj)
  endforeach()

  add_executable(all_tests)
  target_link_libraries(all_tests PRIVATE ${TEST_OBJECTS} Catch2::Catch2WithMain)
endif()

target_include_directories(all_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(all_tests)
