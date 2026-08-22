

# Struct dynampi::NaiveWorkDistributor::Config



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**NaiveWorkDistributor**](classdynampi_1_1NaiveWorkDistributor.md) **>** [**Config**](structdynampi_1_1NaiveWorkDistributor_1_1Config.md)





* `#include <naive_distributor.hpp>`





















## Public Attributes

| Type | Name |
| ---: | :--- |
|  [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**auto\_run\_workers**](#variable-auto_run_workers)   = `[**true**](namespacedynampi.md#function-check_fixed_size_mpi_type)`<br> |
|  [**MPI\_Comm**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**comm**](#variable-comm)   = `[**MPI\_COMM\_WORLD**](namespacedynampi.md#function-check_fixed_size_mpi_type)`<br> |
|  [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**manager\_rank**](#variable-manager_rank)   = `0`<br> |
|  [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**max\_result\_size**](#variable-max_result_size)   = `1024`<br> |
|  [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**rethrow\_task\_errors**](#variable-rethrow_task_errors)   = `[**true**](namespacedynampi.md#function-check_fixed_size_mpi_type)`<br> |
|  [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**use\_immediate\_recv**](#variable-use_immediate_recv)   = `[**false**](namespacedynampi.md#function-check_fixed_size_mpi_type)`<br> |












































## Public Attributes Documentation




### variable auto\_run\_workers 

```C++
bool dynampi::NaiveWorkDistributor< TaskT, ResultT, Options >::Config::auto_run_workers;
```




<hr>



### variable comm 

```C++
MPI_Comm dynampi::NaiveWorkDistributor< TaskT, ResultT, Options >::Config::comm;
```




<hr>



### variable manager\_rank 

```C++
int dynampi::NaiveWorkDistributor< TaskT, ResultT, Options >::Config::manager_rank;
```




<hr>



### variable max\_result\_size 

```C++
int dynampi::NaiveWorkDistributor< TaskT, ResultT, Options >::Config::max_result_size;
```




<hr>



### variable rethrow\_task\_errors 

```C++
bool dynampi::NaiveWorkDistributor< TaskT, ResultT, Options >::Config::rethrow_task_errors;
```




<hr>



### variable use\_immediate\_recv 

```C++
bool dynampi::NaiveWorkDistributor< TaskT, ResultT, Options >::Config::use_immediate_recv;
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/naive_distributor.hpp`

