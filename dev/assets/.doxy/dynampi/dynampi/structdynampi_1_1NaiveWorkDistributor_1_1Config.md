

# Struct dynampi::NaiveWorkDistributor::Config



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**NaiveWorkDistributor**](classdynampi_1_1NaiveWorkDistributor.md) **>** [**Config**](structdynampi_1_1NaiveWorkDistributor_1_1Config.md)





* `#include <naive_distributor.hpp>`





















## Public Attributes

| Type | Name |
| ---: | :--- |
|  [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**auto\_run\_workers**](#variable-auto_run_workers)   = `[**true**](namespacedynampi.md#function-check_fixed_size_mpi_type)`<br> |
|  [**MPI\_Comm**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**comm**](#variable-comm)   = `[**MPI\_COMM\_WORLD**](namespacedynampi.md#function-check_fixed_size_mpi_type)`<br> |
|  [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**manager\_rank**](#variable-manager_rank)   = `0`<br> |
|  [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**rethrow\_task\_errors**](#variable-rethrow_task_errors)   = `[**true**](namespacedynampi.md#function-check_fixed_size_mpi_type)`<br> |












































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



### variable rethrow\_task\_errors 

```C++
bool dynampi::NaiveWorkDistributor< TaskT, ResultT, Options >::Config::rethrow_task_errors;
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/naive_distributor.hpp`

