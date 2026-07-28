

# File lockfree\_distributor.hpp



[**FileList**](files.md) **>** [**dynampi**](dir_f8db417ebd5c3d89eea80c24e3fd4369.md) **>** [**impl**](dir_87365955bfd9c1869b83a1fbd3fdb221.md) **>** [**lockfree\_distributor.hpp**](lockfree__distributor_8hpp.md)

[Go to the source code of this file](lockfree__distributor_8hpp_source.md)



* `#include <algorithm>`
* `#include <cassert>`
* `#include <chrono>`
* `#include <cstdint>`
* `#include <cstring>`
* `#include <functional>`
* `#include <limits>`
* `#include <map>`
* `#include <optional>`
* `#include <thread>`
* `#include <type_traits>`
* `#include <utility>`
* `#include <vector>`
* `#include "../mpi/mpi_communicator.hpp"`
* `#include "../mpi/mpi_types.hpp"`
* `#include "dynampi/impl/base_distributor.hpp"`
* `#include "dynampi/mpi/mpi_error.hpp"`













## Namespaces

| Type | Name |
| ---: | :--- |
| namespace | [**dynampi**](namespacedynampi.md) <br> |
| namespace | [**detail**](namespacedynampi_1_1detail.md) <br> |


## Classes

| Type | Name |
| ---: | :--- |
| class | [**LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md) &lt;[**typename**](structdynampi_1_1MPI__Type.md) [**TaskT**](structdynampi_1_1MPI__Type.md), [**typename**](structdynampi_1_1MPI__Type.md) [**ResultT**](structdynampi_1_1MPI__Type.md), Options&gt;<br> |
| struct | [**Config**](structdynampi_1_1LockFreeMPIWorkDistributor_1_1Config.md) <br> |
| struct | [**RunConfig**](structdynampi_1_1LockFreeMPIWorkDistributor_1_1RunConfig.md) <br> |
| struct | [**Statistics**](structdynampi_1_1LockFreeMPIWorkDistributor_1_1Statistics.md) <br> |
| class | [**MinimalLockFreeMPIWorkDistributor**](classdynampi_1_1MinimalLockFreeMPIWorkDistributor.md) &lt;[**typename**](structdynampi_1_1MPI__Type.md) [**ResultT**](structdynampi_1_1MPI__Type.md)&gt;<br> |
| struct | [**Config**](structdynampi_1_1MinimalLockFreeMPIWorkDistributor_1_1Config.md) <br> |



















































------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/lockfree_distributor.hpp`

