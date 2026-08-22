

# File hierarchical\_lockfree\_rma\_distributor.hpp



[**FileList**](files.md) **>** [**dynampi**](dir_f8db417ebd5c3d89eea80c24e3fd4369.md) **>** [**impl**](dir_87365955bfd9c1869b83a1fbd3fdb221.md) **>** [**hierarchical\_lockfree\_rma\_distributor.hpp**](hierarchical__lockfree__rma__distributor_8hpp.md)

[Go to the source code of this file](hierarchical__lockfree__rma__distributor_8hpp_source.md)



* `#include <algorithm>`
* `#include <cassert>`
* `#include <cstddef>`
* `#include <cstdint>`
* `#include <cstring>`
* `#include <deque>`
* `#include <functional>`
* `#include <iterator>`
* `#include <limits>`
* `#include <optional>`
* `#include <thread>`
* `#include <vector>`
* `#include "../mpi/mpi_communicator.hpp"`
* `#include "../mpi/mpi_group.hpp"`
* `#include "../mpi/mpi_types.hpp"`
* `#include "dynampi/impl/hierarchical_topology_detail.hpp"`
* `#include "dynampi/impl/rma_detail.hpp"`
* `#include "dynampi/mpi/mpi_error.hpp"`
* `#include "dynampi/task_error.hpp"`
* `#include "dynampi/utilities/timer.hpp"`













## Namespaces

| Type | Name |
| ---: | :--- |
| namespace | [**dynampi**](namespacedynampi.md) <br> |
| namespace | [**detail**](namespacedynampi_1_1detail.md) <br> |


## Classes

| Type | Name |
| ---: | :--- |
| class | [**HierarchicalLockFreeRMAWorkDistributor**](classdynampi_1_1HierarchicalLockFreeRMAWorkDistributor.md) &lt;[**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type), Options&gt;<br> |
| struct | [**Config**](structdynampi_1_1HierarchicalLockFreeRMAWorkDistributor_1_1Config.md) <br> |
| struct | [**RunConfig**](structdynampi_1_1HierarchicalLockFreeRMAWorkDistributor_1_1RunConfig.md) <br> |
| class | [**LockFreeRMALevel**](classdynampi_1_1detail_1_1LockFreeRMALevel.md) &lt;[**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type)&gt;<br> |
| struct | [**ClaimedRange**](structdynampi_1_1detail_1_1LockFreeRMALevel_1_1ClaimedRange.md) <br> |
| struct | [**Config**](structdynampi_1_1detail_1_1LockFreeRMALevel_1_1Config.md) <br> |



















































------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/hierarchical_lockfree_rma_distributor.hpp`

