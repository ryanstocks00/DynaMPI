# SPDX-FileCopyrightText: 2026 Ryan Stocks
# SPDX-License-Identifier: Apache-2.0

# Each example is a standalone MPI program depending only on the dynampi
# interface target -- no test framework, no argument parser.
set(examples
    01_index_tasks
    02_incremental_tasks
    03_ordered_and_prioritized
    04_custom_task_type
    05_lockfree_rma
    06_statistics
    08_task_errors)

foreach(example IN LISTS examples)
  add_executable(${example} ${example}.cpp)
  target_link_libraries(${example} PRIVATE dynampi)
endforeach()
