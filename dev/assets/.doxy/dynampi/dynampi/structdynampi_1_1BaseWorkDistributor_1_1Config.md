

# Struct dynampi::BaseWorkDistributor::Config



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**BaseWorkDistributor**](classdynampi_1_1BaseWorkDistributor.md) **>** [**Config**](structdynampi_1_1BaseWorkDistributor_1_1Config.md)





* `#include <base_distributor.hpp>`





















## Public Attributes

| Type | Name |
| ---: | :--- |
|  [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**auto\_run\_workers**](#variable-auto_run_workers)   = `[**true**](namespacedynampi.md#function-check_fixed_size_mpi_type)`<br> |
|  [**MPI\_Comm**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**comm**](#variable-comm)   = `[**MPI\_COMM\_WORLD**](namespacedynampi.md#function-check_fixed_size_mpi_type)`<br> |
|  [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**manager\_rank**](#variable-manager_rank)   = `0`<br> |












































## Public Attributes Documentation




### variable auto\_run\_workers 

```C++
bool dynampi::BaseWorkDistributor< TaskT, ResultT, Options >::Config::auto_run_workers;
```




<hr>



### variable comm 

```C++
MPI_Comm dynampi::BaseWorkDistributor< TaskT, ResultT, Options >::Config::comm;
```




<hr>



### variable manager\_rank 

```C++
int dynampi::BaseWorkDistributor< TaskT, ResultT, Options >::Config::manager_rank;
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/base_distributor.hpp`

