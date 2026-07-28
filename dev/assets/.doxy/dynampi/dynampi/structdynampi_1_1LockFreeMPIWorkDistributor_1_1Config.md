

# Struct dynampi::LockFreeMPIWorkDistributor::Config



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md) **>** [**Config**](structdynampi_1_1LockFreeMPIWorkDistributor_1_1Config.md)





* `#include <lockfree_distributor.hpp>`





















## Public Attributes

| Type | Name |
| ---: | :--- |
|  bool | [**auto\_run\_workers**](#variable-auto_run_workers)   = `true`<br> |
|  int | [**claim\_batch\_size**](#variable-claim_batch_size)   = `8`<br> |
|  MPI\_Comm | [**comm**](#variable-comm)   = `MPI\_COMM\_WORLD`<br> |
|  int | [**manager\_rank**](#variable-manager_rank)   = `0`<br> |
|  int | [**max\_result\_count**](#variable-max_result_count)   = `256`<br> |
|  int | [**max\_task\_count**](#variable-max_task_count)   = `256`<br> |
|  int | [**max\_tasks**](#variable-max_tasks)   = `8192`<br> |












































## Public Attributes Documentation




### variable auto\_run\_workers 

```C++
bool dynampi::LockFreeMPIWorkDistributor< TaskT, ResultT, Options >::Config::auto_run_workers;
```




<hr>



### variable claim\_batch\_size 

```C++
int dynampi::LockFreeMPIWorkDistributor< TaskT, ResultT, Options >::Config::claim_batch_size;
```




<hr>



### variable comm 

```C++
MPI_Comm dynampi::LockFreeMPIWorkDistributor< TaskT, ResultT, Options >::Config::comm;
```




<hr>



### variable manager\_rank 

```C++
int dynampi::LockFreeMPIWorkDistributor< TaskT, ResultT, Options >::Config::manager_rank;
```




<hr>



### variable max\_result\_count 

```C++
int dynampi::LockFreeMPIWorkDistributor< TaskT, ResultT, Options >::Config::max_result_count;
```




<hr>



### variable max\_task\_count 

```C++
int dynampi::LockFreeMPIWorkDistributor< TaskT, ResultT, Options >::Config::max_task_count;
```




<hr>



### variable max\_tasks 

```C++
int dynampi::LockFreeMPIWorkDistributor< TaskT, ResultT, Options >::Config::max_tasks;
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/lockfree_distributor.hpp`

