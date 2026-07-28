

# Struct dynampi::HierarchicalMPIWorkDistributor::Config



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md) **>** [**Config**](structdynampi_1_1HierarchicalMPIWorkDistributor_1_1Config.md)





* `#include <hierarchical_distributor.hpp>`





















## Public Attributes

| Type | Name |
| ---: | :--- |
|  [**bool**](structdynampi_1_1MPI__Type.md) | [**auto\_run\_workers**](#variable-auto_run_workers)   = `[**true**](structdynampi_1_1MPI__Type.md)`<br> |
|  [**int**](structdynampi_1_1MPI__Type.md) | [**batch\_size\_multiplier**](#variable-batch_size_multiplier)   = `2`<br> |
|  [**MPI\_Comm**](structdynampi_1_1MPI__Type.md) | [**comm**](#variable-comm)   = `[**MPI\_COMM\_WORLD**](structdynampi_1_1MPI__Type.md)`<br> |
|  [**bool**](structdynampi_1_1MPI__Type.md) | [**coordinator\_per\_node**](#variable-coordinator_per_node)   = `[**true**](structdynampi_1_1MPI__Type.md)`<br> |
|  [**int**](structdynampi_1_1MPI__Type.md) | [**manager\_rank**](#variable-manager_rank)   = `0`<br> |
|  std::optional&lt; [**int**](structdynampi_1_1MPI__Type.md) &gt; | [**max\_workers\_per\_coordinator**](#variable-max_workers_per_coordinator)   = `std::nullopt`<br> |
|  std::optional&lt; [**size\_t**](structdynampi_1_1MPI__Type.md) &gt; | [**message\_batch\_size**](#variable-message_batch_size)   = `std::nullopt`<br> |












































## Public Attributes Documentation




### variable auto\_run\_workers 

```C++
bool dynampi::HierarchicalMPIWorkDistributor< TaskT, ResultT, Options >::Config::auto_run_workers;
```




<hr>



### variable batch\_size\_multiplier 

```C++
int dynampi::HierarchicalMPIWorkDistributor< TaskT, ResultT, Options >::Config::batch_size_multiplier;
```




<hr>



### variable comm 

```C++
MPI_Comm dynampi::HierarchicalMPIWorkDistributor< TaskT, ResultT, Options >::Config::comm;
```




<hr>



### variable coordinator\_per\_node 

```C++
bool dynampi::HierarchicalMPIWorkDistributor< TaskT, ResultT, Options >::Config::coordinator_per_node;
```




<hr>



### variable manager\_rank 

```C++
int dynampi::HierarchicalMPIWorkDistributor< TaskT, ResultT, Options >::Config::manager_rank;
```




<hr>



### variable max\_workers\_per\_coordinator 

```C++
std::optional<int> dynampi::HierarchicalMPIWorkDistributor< TaskT, ResultT, Options >::Config::max_workers_per_coordinator;
```




<hr>



### variable message\_batch\_size 

```C++
std::optional<size_t> dynampi::HierarchicalMPIWorkDistributor< TaskT, ResultT, Options >::Config::message_batch_size;
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/hierarchical_distributor.hpp`

