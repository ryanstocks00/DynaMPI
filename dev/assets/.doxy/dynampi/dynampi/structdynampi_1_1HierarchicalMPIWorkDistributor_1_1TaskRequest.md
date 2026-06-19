

# Struct dynampi::HierarchicalMPIWorkDistributor::TaskRequest



[**ClassList**](annotated.md) **>** [**TaskRequest**](structdynampi_1_1HierarchicalMPIWorkDistributor_1_1TaskRequest.md)


























## Public Attributes

| Type | Name |
| ---: | :--- |
|  std::optional&lt; [**int**](structdynampi_1_1MPI__Type.md) &gt; | [**num\_tasks\_requested**](#variable-num_tasks_requested)   = `std::nullopt`<br> |
|  CommLayer | [**source\_layer**](#variable-source_layer)   = `CommLayer::Global`<br> |
|  [**int**](structdynampi_1_1MPI__Type.md) | [**worker\_rank**](#variable-worker_rank)  <br> |












































## Public Attributes Documentation




### variable num\_tasks\_requested 

```C++
std::optional<int> dynampi::HierarchicalMPIWorkDistributor< TaskT, ResultT, Options >::TaskRequest::num_tasks_requested;
```




<hr>



### variable source\_layer 

```C++
CommLayer dynampi::HierarchicalMPIWorkDistributor< TaskT, ResultT, Options >::TaskRequest::source_layer;
```




<hr>



### variable worker\_rank 

```C++
int dynampi::HierarchicalMPIWorkDistributor< TaskT, ResultT, Options >::TaskRequest::worker_rank;
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/hierarchical_distributor.hpp`

