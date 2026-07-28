

# Class dynampi::BaseMPIWorkDistributor

**template &lt;[**typename**](structdynampi_1_1MPI__Type.md) [**TaskT**](structdynampi_1_1MPI__Type.md), [**typename**](structdynampi_1_1MPI__Type.md) [**ResultT**](structdynampi_1_1MPI__Type.md), typename... Options&gt;**



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**BaseMPIWorkDistributor**](classdynampi_1_1BaseMPIWorkDistributor.md)





* `#include <base_distributor.hpp>`















## Classes

| Type | Name |
| ---: | :--- |
| struct | [**Config**](structdynampi_1_1BaseMPIWorkDistributor_1_1Config.md) <br> |


























## Protected Types

| Type | Name |
| ---: | :--- |
| typedef std::conditional\_t&lt; [**prioritize\_tasks**](classdynampi_1_1BaseMPIWorkDistributor.md#variable-prioritize_tasks), std::priority\_queue&lt; std::pair&lt; [**double**](structdynampi_1_1MPI__Type.md), [**TaskT**](structdynampi_1_1MPI__Type.md) &gt; &gt;, std::deque&lt; [**TaskT**](structdynampi_1_1MPI__Type.md) &gt; &gt; | [**QueueT**](#typedef-queuet)  <br> |






## Protected Static Attributes

| Type | Name |
| ---: | :--- |
|  [**constexpr**](structdynampi_1_1MPI__Type.md) [**bool**](structdynampi_1_1MPI__Type.md) | [**prioritize\_tasks**](#variable-prioritize_tasks)   = `[**get\_option\_value**](template__options_8hpp.md#function-get_option_value)&lt;[**prioritize\_tasks\_t**](structdynampi_1_1prioritize__tasks__t.md), Options...&gt;()`<br> |


















## Protected Types Documentation




### typedef QueueT 

```C++
using dynampi::BaseMPIWorkDistributor< TaskT, ResultT, Options >::QueueT =  std::conditional_t<prioritize_tasks, std::priority_queue<std::pair<double, TaskT> >, std::deque<TaskT> >;
```




<hr>
## Protected Static Attributes Documentation




### variable prioritize\_tasks 

```C++
constexpr bool dynampi::BaseMPIWorkDistributor< TaskT, ResultT, Options >::prioritize_tasks;
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/base_distributor.hpp`

