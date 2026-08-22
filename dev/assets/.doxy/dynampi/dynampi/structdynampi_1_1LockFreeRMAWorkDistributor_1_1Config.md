

# Struct dynampi::LockFreeRMAWorkDistributor::Config



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**LockFreeRMAWorkDistributor**](classdynampi_1_1LockFreeRMAWorkDistributor.md) **>** [**Config**](structdynampi_1_1LockFreeRMAWorkDistributor_1_1Config.md)





* `#include <lockfree_rma_distributor.hpp>`





















## Public Attributes

| Type | Name |
| ---: | :--- |
|  [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**auto\_run\_workers**](#variable-auto_run_workers)   = `[**true**](namespacedynampi.md#function-check_fixed_size_mpi_type)`<br> |
|  [**MPI\_Comm**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**comm**](#variable-comm)   = `[**MPI\_COMM\_WORLD**](namespacedynampi.md#function-check_fixed_size_mpi_type)`<br> |
|  [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**manager\_rank**](#variable-manager_rank)   = `0`<br> |
|  [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**max\_result\_count**](#variable-max_result_count)   = `256`<br> |
|  [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**max\_task\_count**](#variable-max_task_count)   = `256`<br> |
|  [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**max\_tasks**](#variable-max_tasks)   = `8192`<br> |
|  [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**rethrow\_task\_errors**](#variable-rethrow_task_errors)   = `[**true**](namespacedynampi.md#function-check_fixed_size_mpi_type)`<br> |












































## Public Attributes Documentation




### variable auto\_run\_workers 

```C++
bool dynampi::LockFreeRMAWorkDistributor< TaskT, ResultT, Options >::Config::auto_run_workers;
```




<hr>



### variable comm 

```C++
MPI_Comm dynampi::LockFreeRMAWorkDistributor< TaskT, ResultT, Options >::Config::comm;
```




<hr>



### variable manager\_rank 

```C++
int dynampi::LockFreeRMAWorkDistributor< TaskT, ResultT, Options >::Config::manager_rank;
```




<hr>



### variable max\_result\_count 

```C++
int dynampi::LockFreeRMAWorkDistributor< TaskT, ResultT, Options >::Config::max_result_count;
```




<hr>



### variable max\_task\_count 

```C++
int dynampi::LockFreeRMAWorkDistributor< TaskT, ResultT, Options >::Config::max_task_count;
```




<hr>



### variable max\_tasks 

```C++
int dynampi::LockFreeRMAWorkDistributor< TaskT, ResultT, Options >::Config::max_tasks;
```




<hr>



### variable rethrow\_task\_errors 

```C++
bool dynampi::LockFreeRMAWorkDistributor< TaskT, ResultT, Options >::Config::rethrow_task_errors;
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/lockfree_rma_distributor.hpp`

