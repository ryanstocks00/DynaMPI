# SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
# SPDX-License-Identifier: MIT

enable_language(C)

# Create the C interface library
add_library(dynampi_c SHARED
    src/dynampi.cpp
)

# Set the target properties
set_target_properties(dynampi_c PROPERTIES
    OUTPUT_NAME "dynampi_c"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "include/dynampi.h"
)

# Include directories
target_include_directories(dynampi_c
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link against the main dynampi library and MPI
target_link_libraries(dynampi_c
    PRIVATE
        dynampi
        MPI::MPI_CXX
)

# Set C++ standard
target_compile_features(dynampi_c PRIVATE cxx_std_20)

# Compiler flags
if(MSVC)
    target_compile_options(dynampi_c PRIVATE /W4 /WX)
else()
    target_compile_options(dynampi_c PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

# Add test executable
if(DYNAMPI_BUILD_TESTS)
    add_executable(c_interface_test test/test_c_interface.c)
    target_link_libraries(c_interface_test PRIVATE dynampi_c MPI::MPI_CXX)
    target_include_directories(c_interface_test PRIVATE include)

    # Detect oversubscribe settings used for MPI tests
    include(mpi_oversubscribe)

    # Add MPI test
    add_test(NAME c_interface_test_1_rank COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPI_OVERSUBSCRIBE_FLAG} $<TARGET_FILE:c_interface_test>)
    set_tests_properties(c_interface_test_1_rank PROPERTIES TIMEOUT 10)
    add_test(NAME c_interface_test_2_rank COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPI_OVERSUBSCRIBE_FLAG} $<TARGET_FILE:c_interface_test>)
    set_tests_properties(c_interface_test_2_rank PROPERTIES TIMEOUT 10)
    add_test(NAME c_interface_test_4_rank COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPI_OVERSUBSCRIBE_FLAG} $<TARGET_FILE:c_interface_test>)
    set_tests_properties(c_interface_test_4_rank PROPERTIES TIMEOUT 10)
endif()

# Install rules
install(TARGETS dynampi_c
    EXPORT DynaMPICTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include
)

# Export targets
install(EXPORT DynaMPICTargets
    FILE DynaMPICTargets.cmake
    NAMESPACE DynaMPI::
    DESTINATION lib/cmake/DynaMPIC
)

# Create and install config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/DynaMPICConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/DynaMPICConfig.cmake"
    INSTALL_DESTINATION lib/cmake/DynaMPIC
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/DynaMPICConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/DynaMPICConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/DynaMPICConfigVersion.cmake"
    DESTINATION lib/cmake/DynaMPIC
)
