

# Struct dynampi::NaiveWorkDistributor::Statistics



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**NaiveWorkDistributor**](classdynampi_1_1NaiveWorkDistributor.md) **>** [**Statistics**](structdynampi_1_1NaiveWorkDistributor_1_1Statistics.md)





* `#include <naive_distributor.hpp>`





















## Public Attributes

| Type | Name |
| ---: | :--- |
|  [**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**CommStatistics**](structdynampi_1_1CommStatistics.md) & | [**comm\_statistics**](#variable-comm_statistics)  <br> |
|  std::vector&lt; [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; | [**worker\_task\_counts**](#variable-worker_task_counts)  <br> |












































## Public Attributes Documentation




### variable comm\_statistics 

```C++
const CommStatistics& dynampi::NaiveWorkDistributor< TaskT, ResultT, Options >::Statistics::comm_statistics;
```




<hr>



### variable worker\_task\_counts 

```C++
std::vector<size_t> dynampi::NaiveWorkDistributor< TaskT, ResultT, Options >::Statistics::worker_task_counts;
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/naive_distributor.hpp`

