

# Class dynampi::BaseWorkDistributor

**template &lt;[**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type), typename... Options&gt;**



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**BaseWorkDistributor**](classdynampi_1_1BaseWorkDistributor.md)





* `#include <base_distributor.hpp>`















## Classes

| Type | Name |
| ---: | :--- |
| struct | [**Config**](structdynampi_1_1BaseWorkDistributor_1_1Config.md) <br> |


























## Protected Types

| Type | Name |
| ---: | :--- |
| typedef std::conditional\_t&lt; [**prioritize\_tasks**](classdynampi_1_1BaseWorkDistributor.md#variable-prioritize_tasks), std::priority\_queue&lt; std::pair&lt; [**double**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; &gt;, std::deque&lt; [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; &gt; | [**QueueT**](#typedef-queuet)  <br> |






## Protected Static Attributes

| Type | Name |
| ---: | :--- |
|  [**constexpr**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**prioritize\_tasks**](#variable-prioritize_tasks)   = `[**get\_option\_value**](template__options_8hpp.md#function-get_option_value)&lt;[**prioritize\_tasks\_t**](structdynampi_1_1prioritize__tasks__t.md), Options...&gt;()`<br> |


















## Protected Types Documentation




### typedef QueueT 

```C++
using dynampi::BaseWorkDistributor< TaskT, ResultT, Options >::QueueT =  std::conditional_t<prioritize_tasks, std::priority_queue<std::pair<double, TaskT> >, std::deque<TaskT> >;
```




<hr>
## Protected Static Attributes Documentation




### variable prioritize\_tasks 

```C++
constexpr bool dynampi::BaseWorkDistributor< TaskT, ResultT, Options >::prioritize_tasks;
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/base_distributor.hpp`

