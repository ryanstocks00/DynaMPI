

# Struct dynampi::HierarchicalWorkDistributor::Config



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**HierarchicalWorkDistributor**](classdynampi_1_1HierarchicalWorkDistributor.md) **>** [**Config**](structdynampi_1_1HierarchicalWorkDistributor_1_1Config.md)





* `#include <hierarchical_distributor.hpp>`





















## Public Attributes

| Type | Name |
| ---: | :--- |
|  [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**auto\_run\_workers**](#variable-auto_run_workers)   = `[**true**](namespacedynampi.md#function-check_fixed_size_mpi_type)`<br> |
|  [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**batch\_size\_multiplier**](#variable-batch_size_multiplier)   = `1`<br> |
|  [**MPI\_Comm**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**comm**](#variable-comm)   = `[**MPI\_COMM\_WORLD**](namespacedynampi.md#function-check_fixed_size_mpi_type)`<br> |
|  [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**manager\_per\_node**](#variable-manager_per_node)   = `[**true**](namespacedynampi.md#function-check_fixed_size_mpi_type)`<br> |
|  [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**manager\_rank**](#variable-manager_rank)   = `0`<br> |
|  [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**max\_local\_group\_size**](#variable-max_local_group_size)   = `0`<br> |
|  [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**max\_upper\_fanout**](#variable-max_upper_fanout)   = `-1`<br> |
|  std::optional&lt; [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; | [**max\_workers\_per\_manager**](#variable-max_workers_per_manager)   = `std::nullopt`<br> |
|  [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**pipeline\_depth**](#variable-pipeline_depth)   = `2`<br> |
|  [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**rethrow\_task\_errors**](#variable-rethrow_task_errors)   = `[**true**](namespacedynampi.md#function-check_fixed_size_mpi_type)`<br> |












































## Public Attributes Documentation




### variable auto\_run\_workers 

```C++
bool dynampi::HierarchicalWorkDistributor< TaskT, ResultT, Options >::Config::auto_run_workers;
```




<hr>



### variable batch\_size\_multiplier 

```C++
int dynampi::HierarchicalWorkDistributor< TaskT, ResultT, Options >::Config::batch_size_multiplier;
```




<hr>



### variable comm 

```C++
MPI_Comm dynampi::HierarchicalWorkDistributor< TaskT, ResultT, Options >::Config::comm;
```




<hr>



### variable manager\_per\_node 

```C++
bool dynampi::HierarchicalWorkDistributor< TaskT, ResultT, Options >::Config::manager_per_node;
```




<hr>



### variable manager\_rank 

```C++
int dynampi::HierarchicalWorkDistributor< TaskT, ResultT, Options >::Config::manager_rank;
```




<hr>



### variable max\_local\_group\_size 

```C++
int dynampi::HierarchicalWorkDistributor< TaskT, ResultT, Options >::Config::max_local_group_size;
```




<hr>



### variable max\_upper\_fanout 

```C++
int dynampi::HierarchicalWorkDistributor< TaskT, ResultT, Options >::Config::max_upper_fanout;
```




<hr>



### variable max\_workers\_per\_manager 

```C++
std::optional<int> dynampi::HierarchicalWorkDistributor< TaskT, ResultT, Options >::Config::max_workers_per_manager;
```




<hr>



### variable pipeline\_depth 

```C++
int dynampi::HierarchicalWorkDistributor< TaskT, ResultT, Options >::Config::pipeline_depth;
```




<hr>



### variable rethrow\_task\_errors 

```C++
bool dynampi::HierarchicalWorkDistributor< TaskT, ResultT, Options >::Config::rethrow_task_errors;
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/hierarchical_distributor.hpp`

