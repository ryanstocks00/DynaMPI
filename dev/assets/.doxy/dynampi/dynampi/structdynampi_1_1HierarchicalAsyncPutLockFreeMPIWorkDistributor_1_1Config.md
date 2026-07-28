

# Struct dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor::Config



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**HierarchicalAsyncPutLockFreeMPIWorkDistributor**](classdynampi_1_1HierarchicalAsyncPutLockFreeMPIWorkDistributor.md) **>** [**Config**](structdynampi_1_1HierarchicalAsyncPutLockFreeMPIWorkDistributor_1_1Config.md)





* `#include <hierarchical_async_put_lockfree_distributor.hpp>`





















## Public Attributes

| Type | Name |
| ---: | :--- |
|  bool | [**auto\_run\_workers**](#variable-auto_run_workers)   = `true`<br> |
|  MPI\_Comm | [**comm**](#variable-comm)   = `MPI\_COMM\_WORLD`<br> |
|  int | [**leader\_batch\_multiplier**](#variable-leader_batch_multiplier)   = `2`<br> |
|  int | [**local\_batch\_size**](#variable-local_batch_size)   = `8`<br> |
|  int | [**manager\_rank**](#variable-manager_rank)   = `0`<br> |
|  int | [**max\_local\_tasks**](#variable-max_local_tasks)   = `8192`<br> |
|  int | [**max\_result\_count**](#variable-max_result_count)   = `256`<br> |
|  int | [**max\_task\_count**](#variable-max_task_count)   = `256`<br> |
|  int | [**max\_tasks**](#variable-max_tasks)   = `8192`<br> |
|  int | [**max\_upper\_fanout**](#variable-max_upper_fanout)   = `0`<br> |












































## Public Attributes Documentation




### variable auto\_run\_workers 

```C++
bool dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor< TaskT, ResultT, Options >::Config::auto_run_workers;
```




<hr>



### variable comm 

```C++
MPI_Comm dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor< TaskT, ResultT, Options >::Config::comm;
```




<hr>



### variable leader\_batch\_multiplier 

```C++
int dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor< TaskT, ResultT, Options >::Config::leader_batch_multiplier;
```




<hr>



### variable local\_batch\_size 

```C++
int dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor< TaskT, ResultT, Options >::Config::local_batch_size;
```




<hr>



### variable manager\_rank 

```C++
int dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor< TaskT, ResultT, Options >::Config::manager_rank;
```




<hr>



### variable max\_local\_tasks 

```C++
int dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor< TaskT, ResultT, Options >::Config::max_local_tasks;
```




<hr>



### variable max\_result\_count 

```C++
int dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor< TaskT, ResultT, Options >::Config::max_result_count;
```




<hr>



### variable max\_task\_count 

```C++
int dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor< TaskT, ResultT, Options >::Config::max_task_count;
```




<hr>



### variable max\_tasks 

```C++
int dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor< TaskT, ResultT, Options >::Config::max_tasks;
```




<hr>



### variable max\_upper\_fanout 

```C++
int dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor< TaskT, ResultT, Options >::Config::max_upper_fanout;
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/hierarchical_async_put_lockfree_distributor.hpp`

