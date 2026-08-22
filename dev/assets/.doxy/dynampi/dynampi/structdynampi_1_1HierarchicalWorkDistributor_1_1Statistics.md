

# Struct dynampi::HierarchicalWorkDistributor::Statistics



[**ClassList**](annotated.md) **>** [**Statistics**](structdynampi_1_1HierarchicalWorkDistributor_1_1Statistics.md)


























## Public Attributes

| Type | Name |
| ---: | :--- |
|  [**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**CommStatistics**](structdynampi_1_1CommStatistics.md) & | [**comm\_statistics**](#variable-comm_statistics)  <br> |
|  std::optional&lt; std::vector&lt; [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; &gt; | [**worker\_task\_counts**](#variable-worker_task_counts)   = `{}`<br> |












































## Public Attributes Documentation




### variable comm\_statistics 

```C++
const CommStatistics& dynampi::HierarchicalWorkDistributor< TaskT, ResultT, Options >::Statistics::comm_statistics;
```




<hr>



### variable worker\_task\_counts 

```C++
std::optional<std::vector<size_t> > dynampi::HierarchicalWorkDistributor< TaskT, ResultT, Options >::Statistics::worker_task_counts;
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/hierarchical_distributor.hpp`

