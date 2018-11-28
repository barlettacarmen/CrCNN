cmake_minimum_required(VERSION 3.10)

project(SEAL VERSION 2.3.1 LANGUAGES CXX C)

if(${MSVC})
    message(FATAL_ERROR "Please build SEAL using the attached Visual Studio solution/project files.")
endif()

# Build in Release mode by default; otherwise use selected option
set(SEAL_DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE ${SEAL_DEFAULT_BUILD_TYPE} CACHE
        STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Release" "Debug" "MinSizeRel" "RelWithDebInfo")
endif()
message(STATUS "Build type (CMAKE_BUILD_TYPE): ${CMAKE_BUILD_TYPE}")

# In Debug mode enable also SEAL_DEBUG by default
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set(SEAL_DEBUG_DEFAULT ON)
else()
    set(SEAL_DEBUG_DEFAULT OFF)
endif()

# Required files and directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${SEAL_SOURCE_DIR}/../lib)
set(SEAL_INCLUDES_INSTALL_DIR include)
set(SEAL_CONFIG_IN_FILENAME ${SEAL_SOURCE_DIR}/cmake/SEALConfig.cmake.in)
set(SEAL_CONFIG_FILENAME ${SEAL_SOURCE_DIR}/cmake/SEALConfig.cmake)
set(SEAL_TARGETS_FILENAME ${SEAL_SOURCE_DIR}/cmake/SEALTargets.cmake)
set(SEAL_CONFIG_VERSION_FILENAME ${SEAL_SOURCE_DIR}/cmake/SEALConfigVersion.cmake)
set(SEAL_CONFIG_INSTALL_DIR lib/cmake/SEAL)

# For extra modules we might have 
list(APPEND CMAKE_MODULE_PATH ${SEAL_SOURCE_DIR}/cmake)

include(CMakePushCheckState)
include(CMakeDependentOption)
include(CheckIncludeFiles)
include(CheckCXXSourceRuns)
include(CheckTypeSize)

# Are we using SEAL_DEBUG?
set(SEAL_DEBUG ${SEAL_DEBUG_DEFAULT})
message(STATUS "SEAL debug mode: ${SEAL_DEBUG}")

# Use intrinsics if available
set(SEAL_USE_INTRIN_OPTION_STR "Use intrinsics")
option(SEAL_USE_INTRIN ${SEAL_USE_INTRIN_OPTION_STR} ON)

# Use Microsoft GSL if available
set(SEAL_USE_MSGSL_OPTION_STR "Use Microsoft GSL (experimental)")
option(SEAL_USE_MSGSL ${SEAL_USE_MSGSL_OPTION_STR} OFF)

# Check for x64intrin.h
if(SEAL_USE_INTRIN)
    check_include_file("x86intrin.h" HAVE_INTRIN_HEADER)
    if(NOT HAVE_INTRIN_HEADER)
        set(SEAL_USE_INTRIN OFF CACHE BOOL ${SEAL_USE_INTRIN_OPTION_STR} FORCE)
    endif()
endif()
message(STATUS "${SEAL_USE_INTRIN_OPTION_STR} (SEAL_USE_INTRIN): ${SEAL_USE_INTRIN}")

# Specific intrinsics depending on SEAL_USE_INTRIN
set(SEAL_USE___INT128_OPTION_STR "Use __int128")
cmake_dependent_option(SEAL_USE___INT128 ${SEAL_USE___INT128_OPTION_STR} ON "SEAL_USE_INTRIN" OFF)

set(SEAL_USE___BUILTIN_CLZLL_OPTION_STR "Use __builtin_clzll")
cmake_dependent_option(SEAL_USE___BUILTIN_CLZLL ${SEAL_USE___BUILTIN_CLZLL_OPTION_STR} ON "SEAL_USE_INTRIN" OFF)

set(SEAL_USE__ADDCARRY_U64_OPTION_STR "Use _addcarry_u64")
cmake_dependent_option(SEAL_USE__ADDCARRY_U64 ${SEAL_USE__ADDCARRY_U64_OPTION_STR} ON "SEAL_USE_INTRIN" OFF)

set(SEAL_USE__SUBBORROW_U64_OPTION_STR "Use _subborrow_u64")
cmake_dependent_option(SEAL_USE__SUBBORROW_U64 ${SEAL_USE__SUBBORROW_U64_OPTION_STR} ON "SEAL_USE_INTRIN" OFF)

# Specific options depending on SEAL_USE_MSGSL
set(SEAL_USE_MSGSL_SPAN_OPTION_STR "Use gsl::span")
cmake_dependent_option(SEAL_USE_MSGSL_SPAN ${SEAL_USE_MSGSL_SPAN_OPTION_STR} ON "SEAL_USE_MSGSL" OFF)

set(SEAL_USE_MSGSL_MULTISPAN_OPTION_STR "Use gsl::multi_span")
cmake_dependent_option(SEAL_USE_MSGSL_MULTISPAN ${SEAL_USE_MSGSL_MULTISPAN_OPTION_STR} ON "SEAL_USE_MSGSL" OFF)

if(SEAL_USE_INTRIN)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_FLAGS "-O0")
    set(CMAKE_REQUIRED_QUIET TRUE)

    # Check for presence of ___int128
    if(SEAL_USE___INT128)
        check_type_size("__int128" INT128 LANGUAGE CXX)
        if(NOT INT128 EQUAL 16)
            set(SEAL_USE___INT128 OFF CACHE BOOL ${SEAL_USE___INT128_OPTION_STR} FORCE)
        endif()
    endif()
    message(STATUS "${SEAL_USE___INT128_OPTION_STR}: ${SEAL_USE___INT128}")

    # Check for __builtin_clzll
    if(SEAL_USE___BUILTIN_CLZLL)
        check_cxx_source_runs("
            #include <x86intrin.h> 
            int main() { 
                __builtin_clzll(0); 
                return 0; 
            }"
            USE_BUILTIN_CLZLL)
        if(NOT USE_BUILTIN_CLZLL EQUAL 1)
            set(SEAL_USE___BUILTIN_CLZLL OFF CACHE BOOL ${SEAL_USE___BUILTIN_CLZLL_OPTION_STR} FORCE)
        endif()
    endif()
    message(STATUS "${SEAL_USE___BUILTIN_CLZLL_OPTION_STR}: ${SEAL_USE___BUILTIN_CLZLL}")

    # Check for _addcarry_u64
    if(SEAL_USE__ADDCARRY_U64)
        check_cxx_source_runs("
            #include <x86intrin.h> 
            int main() { 
                unsigned long long a;
                _addcarry_u64(0,0,0,&a); 
                return 0; 
            }"
            USE_ADDCARRY_U64)
        if(NOT USE_ADDCARRY_U64 EQUAL 1)
            set(SEAL_USE__ADDCARRY_U64 OFF CACHE BOOL ${SEAL_USE__ADDCARRY_U64_OPTION_STR} FORCE)
        endif()
    endif()
    message(STATUS "${SEAL_USE__ADDCARRY_U64_OPTION_STR}: ${SEAL_USE__ADDCARRY_U64}")

    # Check for _subborrow_u64
    if(SEAL_USE__SUBBORROW_U64)
        check_cxx_source_runs("
            #include <x86intrin.h> 
            int main() { 
                unsigned long long a;
                _subborrow_u64(0,0,0,&a); 
                return 0; 
            }"
            USE_SUBBORROW_U64)
        if(NOT USE_SUBBORROW_U64 EQUAL 1)
            set(SEAL_USE__SUBBORROW_U64 OFF CACHE BOOL ${SEAL_USE__SUBBORROW_U64_OPTION_STR} FORCE)
        endif()
    endif()
    message(STATUS "${SEAL_USE__SUBBORROW_U64_OPTION_STR}: ${SEAL_USE__SUBBORROW_U64}")

    cmake_pop_check_state()
endif()

# Try to find MSGSL if requested
if(SEAL_USE_MSGSL)
    find_package(msgsl MODULE)
    if(NOT msgsl_FOUND)
        set(SEAL_USE_MSGSL OFF CACHE BOOL ${SEAL_USE_MSGSL_OPTION_STR} FORCE)
    endif()
endif()
message(STATUS "${SEAL_USE_MSGSL_OPTION_STR} (SEAL_USE_MSGSL): ${SEAL_USE_MSGSL}")

if(SEAL_USE_MSGSL)
    # Now check for individual classes
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_INCLUDES ${MSGSL_INCLUDE_DIR})
    set(CMAKE_EXTRA_INCLUDE_FILES gsl/gsl)
    set(CMAKE_REQUIRED_QUIET TRUE)

    # Detect gsl::span
    if(SEAL_USE_MSGSL_SPAN)
        check_type_size("gsl::span<std::uint64_t>" MSGSL_SPAN LANGUAGE CXX)
        if(NOT MSGSL_SPAN GREATER 0)
            set(SEAL_USE_MSGSL_SPAN OFF CACHE BOOL ${SEAL_USE_MSGSL_SPAN_OPTION_STR} FORCE)
        endif()
    endif()
    message(STATUS "${SEAL_USE_MSGSL_SPAN_OPTION_STR}: ${SEAL_USE_MSGSL_SPAN}")

    # Detect gsl::multi_span
    if(SEAL_USE_MSGSL_MULTISPAN)
        check_type_size("gsl::multi_span<std::uint64_t, 1, gsl::dynamic_range>" MSGSL_MULTISPAN LANGUAGE CXX)
        if(NOT MSGSL_MULTISPAN GREATER 0)
            set(SEAL_USE_MSGSL_MULTISPAN OFF CACHE BOOL ${SEAL_USE_MSGSL_MULTISPAN_OPTION_STR} FORCE)
        endif()
    endif()
    message(STATUS "${SEAL_USE_MSGSL_MULTISPAN_OPTION_STR}: ${SEAL_USE_MSGSL_MULTISPAN}")

    cmake_pop_check_state()
endif() 

# Create library but add no source files yet
add_library(seal STATIC "")

# Add source files to library and header files to install
add_subdirectory(seal)

# Add local include directories for build
target_include_directories(seal PUBLIC 
    $<BUILD_INTERFACE:${SEAL_SOURCE_DIR}>)

# Require C++17
target_compile_features(seal PUBLIC cxx_std_17)

# Require thread library
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

# Link Threads::Threads with seal
target_link_libraries(seal PUBLIC Threads::Threads)

# Create msgsl interface target
if(SEAL_USE_MSGSL)
    # Create interface target
    add_library(msgsl INTERFACE)
    set_target_properties(msgsl PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${MSGSL_INCLUDE_DIR})

    # Associate msgsl with export seal_export
    install(TARGETS msgsl EXPORT seal_export)

    # Link with seal
    target_link_libraries(seal PUBLIC msgsl)
endif()

# Associate seal to export seal_export
install(TARGETS seal EXPORT seal_export
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION ${SEAL_INCLUDES_INSTALL_DIR})

# Export the targets
export(EXPORT seal_export
    FILE ${SEAL_TARGETS_FILENAME}
    NAMESPACE SEAL::)

# Create the CMake config file
configure_file(${SEAL_CONFIG_IN_FILENAME} ${SEAL_CONFIG_FILENAME} @ONLY)

# Install the export
install(
    EXPORT seal_export
    FILE SEALTargets.cmake
    NAMESPACE SEAL::
    DESTINATION ${SEAL_CONFIG_INSTALL_DIR})

# Version file; we require exact version match for downstream
include(CMakePackageConfigHelpers)
write_basic_package_version_file(${SEAL_CONFIG_VERSION_FILENAME}
    VERSION ${SEAL_VERSION}
    COMPATIBILITY ExactVersion)

# Install other files 
install(
    FILES
        ${SEAL_CONFIG_FILENAME}
        ${SEAL_CONFIG_VERSION_FILENAME}
    DESTINATION ${SEAL_CONFIG_INSTALL_DIR})
