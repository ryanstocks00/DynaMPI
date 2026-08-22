

# Struct dynampi::detail::LockFreeRMALevel::Config



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**detail**](namespacedynampi_1_1detail.md) **>** [**LockFreeRMALevel**](classdynampi_1_1detail_1_1LockFreeRMALevel.md) **>** [**Config**](structdynampi_1_1detail_1_1LockFreeRMALevel_1_1Config.md)





* `#include <hierarchical_lockfree_rma_distributor.hpp>`





















## Public Attributes

| Type | Name |
| ---: | :--- |
|  [**MPI\_Comm**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**comm**](#variable-comm)   = `[**MPI\_COMM\_NULL**](namespacedynampi.md#function-check_fixed_size_mpi_type)`<br> |
|  [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**max\_result\_count**](#variable-max_result_count)   = `256`<br> |
|  [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**max\_task\_count**](#variable-max_task_count)   = `256`<br> |
|  [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**max\_tasks**](#variable-max_tasks)   = `8192`<br> |
|  [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**owner\_rank**](#variable-owner_rank)   = `0`<br> |












































## Public Attributes Documentation




### variable comm 

```C++
MPI_Comm dynampi::detail::LockFreeRMALevel< TaskT, ResultT >::Config::comm;
```




<hr>



### variable max\_result\_count 

```C++
int dynampi::detail::LockFreeRMALevel< TaskT, ResultT >::Config::max_result_count;
```




<hr>



### variable max\_task\_count 

```C++
int dynampi::detail::LockFreeRMALevel< TaskT, ResultT >::Config::max_task_count;
```




<hr>



### variable max\_tasks 

```C++
int dynampi::detail::LockFreeRMALevel< TaskT, ResultT >::Config::max_tasks;
```




<hr>



### variable owner\_rank 

```C++
int dynampi::detail::LockFreeRMALevel< TaskT, ResultT >::Config::owner_rank;
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/hierarchical_lockfree_rma_distributor.hpp`

