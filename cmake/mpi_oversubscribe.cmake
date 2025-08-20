# SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
# SPDX-License-Identifier: MIT

# Detect MPI implementation and determine oversubscribe flag if needed

set(MPI_OVERSUBSCRIBE_FLAG "")

if(MPIEXEC_EXECUTABLE)
  execute_process(
    COMMAND ${MPIEXEC_EXECUTABLE} --version
    OUTPUT_VARIABLE MPI_VERSION_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  string(REGEX MATCH "Open MPI|OpenRTE" IS_OPENMPI "${MPI_VERSION_OUTPUT}")

  if(IS_OPENMPI)
    set(MPI_OVERSUBSCRIBE_FLAG "--oversubscribe")
    message(STATUS "Detected OpenMPI. Using oversubscribe flag: '${MPI_OVERSUBSCRIBE_FLAG}'")
  else()
    message(STATUS "MPI implementation detected without explicit oversubscribe requirement")
  endif()

  message(STATUS "MPI version output: ${MPI_VERSION_OUTPUT}")
  message(STATUS "MPI oversubscribe flag: '${MPI_OVERSUBSCRIBE_FLAG}'")
endif()
