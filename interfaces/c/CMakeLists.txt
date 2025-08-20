# SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
# SPDX-License-Identifier: MIT

enable_language(C)

add_library(dynampi_c STATIC
    src/dynampi.cpp
)

# Set the target properties
set_target_properties(dynampi_c PROPERTIES
    OUTPUT_NAME "dynampi_c"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "include/dynampi.h"
    LINKER_LANGUAGE CXX
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

target_link_libraries(dynampi_c
    PUBLIC
        MPI::MPI_CXX
    PRIVATE
        dynampi
)

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

    # Add MPI tests with varying ranks
    function(add_mpi_test test_name num_procs)
    set(command
        ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG}
        ${num_procs}
        ${MPI_OVERSUBSCRIBE_FLAG}
        ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:c_interface_test>
        ${MPIEXEC_POSTFLAGS})
    add_test(NAME ${test_name} COMMAND ${command})
    set_tests_properties(${test_name} PROPERTIES TIMEOUT 30)
    endfunction()

    foreach(rank 1 2 4)
        add_mpi_test(mpi_c_interface_test_${rank}_rank ${rank})
    endforeach()
endif()
