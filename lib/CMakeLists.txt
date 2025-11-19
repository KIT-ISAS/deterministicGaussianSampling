cmake_minimum_required(VERSION 3.10)

project(approxLCD VERSION 1.0 LANGUAGES CXX)

# ensure C++17 or later is used
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# set debug / release flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra" CACHE STRING "Debug build flags" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE
    "-O3 -march=native -ffast-math -funroll-loops -DNDEBUG -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wdouble-promotion -Wformat=2 -Wundef -Wuseless-cast -Wzero-as-null-pointer-constant -Wduplicated-cond -Wduplicated-branches -Wlogical-op"
    CACHE STRING "Release build flags" FORCE)

# lto fix
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND WIN32)
    message(STATUS "MinGW detected. Using LTO-compatible tools.")
    set(CMAKE_AR "gcc-ar")
    set(CMAKE_RANLIB "gcc-ranlib")
endif()

# required packages
find_package(GSL REQUIRED)
find_package(OpenMP REQUIRED)

# dirac_to_dirac sources
set(DIRAC_TO_DIRAC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dirac_to_dirac)
file(GLOB DIRAC_TO_DIRAC_SOURCES ${DIRAC_TO_DIRAC_DIR}/*.cpp)

# gm_to_dirac sources
set(GM_TO_DIRAC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/gm_to_dirac)
file(GLOB GM_TO_DIRAC_SOURCES ${GM_TO_DIRAC_DIR}/*.cpp)

# common sources
set(COMMON_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/debug/capture_time.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gsl_minimizer/gsl_minimizer.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/gsl_multivariative_gradient
)
# common includes
set(COMMON_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/debug
    ${CMAKE_CURRENT_SOURCE_DIR}/gsl_types
    ${CMAKE_CURRENT_SOURCE_DIR}/gsl_minimizer
    ${CMAKE_CURRENT_SOURCE_DIR}/gsl_integration
    ${CMAKE_CURRENT_SOURCE_DIR}/gsl_multivariative_gradient
    ${CMAKE_CURRENT_SOURCE_DIR}/gsl_utils
    ${CMAKE_CURRENT_SOURCE_DIR}/math_utils
    ${CMAKE_CURRENT_SOURCE_DIR}/cache_manager
    ${CMAKE_CURRENT_SOURCE_DIR}/options
)

# static library target
add_library(approxLCD_static STATIC
    ${COMMON_SOURCES}
    ${DIRAC_TO_DIRAC_SOURCES}
    ${GM_TO_DIRAC_SOURCES}
)
target_include_directories(approxLCD_static PUBLIC
    ${COMMON_INCLUDES}
    ${DIRAC_TO_DIRAC_DIR}
    ${GM_TO_DIRAC_DIR}
)
set_target_properties(approxLCD_static PROPERTIES OUTPUT_NAME "approxLCD")
target_link_libraries(approxLCD_static PUBLIC GSL::gsl GSL::gslcblas OpenMP::OpenMP_CXX)

# shared library target
add_library(approxLCD_shared SHARED
    ${COMMON_SOURCES}
    ${DIRAC_TO_DIRAC_SOURCES}
    ${GM_TO_DIRAC_SOURCES}
    )
target_include_directories(approxLCD_shared PUBLIC
    ${COMMON_INCLUDES}
    ${DIRAC_TO_DIRAC_DIR}
    ${GM_TO_DIRAC_DIR}
)

# platform-specific options
# make some includes and linking static
if(UNIX AND NOT APPLE)
    # Linux
    target_compile_options(approxLCD_shared PRIVATE -static-libgcc -static-libstdc++)
    target_link_options(approxLCD_shared PRIVATE -static-libgcc -static-libstdc++)
    set_target_properties(approxLCD_shared PROPERTIES
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
    )
elseif(APPLE)
    # macOS
    target_compile_options(approxLCD_shared PRIVATE -stdlib=libc++)
    set_target_properties(approxLCD_shared PROPERTIES
        BUILD_RPATH "@loader_path"
        INSTALL_RPATH "@loader_path"
    )
elseif(WIN32)
    # Windows MSVC
    if(MSVC)
        target_compile_options(approxLCD_shared PRIVATE /MT)
    else()
        # MinGW
        target_compile_options(approxLCD_shared PRIVATE -static-libgcc -static-libstdc++)
        target_link_options(approxLCD_shared PRIVATE -static-libgcc -static-libstdc++)
    endif()
endif()

set_target_properties(approxLCD_shared PROPERTIES OUTPUT_NAME "approxLCD")
target_link_libraries(approxLCD_shared PUBLIC GSL::gsl GSL::gslcblas OpenMP::OpenMP_CXX)

# GoogleTest / GoogleBenchmark
set(DIRAC_TO_DIRAC_TEST_DIR ${DIRAC_TO_DIRAC_DIR}/tests)
set(GM_TO_DIRAC_TEST_DIR ${GM_TO_DIRAC_DIR}/tests)
set(MATH_UTILS_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/math_utils/tests)

find_package(GTest CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)
enable_testing()

## math utils
file(GLOB MATH_UTILS_UNIT_SOURCES ${MATH_UTILS_TEST_DIR}/unit_tests/*.cpp)
file(GLOB MATH_UTILS_BENCHMARK_SOURCES ${MATH_UTILS_TEST_DIR}/benchmark/*.cpp)

## dirac to dirac
file(GLOB DIRAC_TO_DIRAC_UNIT_SOURCES ${DIRAC_TO_DIRAC_TEST_DIR}/unit_tests/*.cpp)
file(GLOB DIRAC_TO_DIRAC_BENCHMARK_SOURCES ${DIRAC_TO_DIRAC_TEST_DIR}/benchmark/*.cpp)

## gm to dirac
file(GLOB GM_TO_DIRAC_UNIT_SOURCES ${GM_TO_DIRAC_TEST_DIR}/unit_tests/*.cpp)
file(GLOB GM_TO_DIRAC_BENCHMARK_SOURCES ${GM_TO_DIRAC_TEST_DIR}/benchmark/*.cpp)

## unit tests
add_executable(unit_test_target ${DIRAC_TO_DIRAC_UNIT_SOURCES} ${GM_TO_DIRAC_UNIT_SOURCES} ${MATH_UTILS_UNIT_SOURCES})
set_target_properties(unit_test_target PROPERTIES OUTPUT_NAME "unit")
target_include_directories(unit_test_target PRIVATE ${DIRAC_TO_DIRAC_TEST_DIR}/unit_tests ${GM_TO_DIRAC_TEST_DIR}/unit_tests ${MATH_UTILS_TEST_DIR}/unit_tests)
target_link_libraries(unit_test_target PRIVATE 
        GTest::gtest_main
        approxLCD_static)
include(GoogleTest)
gtest_discover_tests(unit_test_target)

## benchmark
add_executable(benchmark_target ${DIRAC_TO_DIRAC_BENCHMARK_SOURCES} ${GM_TO_DIRAC_BENCHMARK_SOURCES} ${MATH_UTILS_BENCHMARK_SOURCES})
set_target_properties(benchmark_target PROPERTIES OUTPUT_NAME "benchmark")
target_link_libraries(benchmark_target PRIVATE 
        benchmark::benchmark 
        benchmark::benchmark_main
        approxLCD_static)
