

# Class dynampi::HierarchicalMPIWorkDistributor

**template &lt;[**typename**](structdynampi_1_1MPI__Type.md) [**TaskT**](structdynampi_1_1MPI__Type.md), [**typename**](structdynampi_1_1MPI__Type.md) [**ResultT**](structdynampi_1_1MPI__Type.md), typename... Options&gt;**



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md)





* `#include <hierarchical_distributor.hpp>`



Inherits the following classes: [dynampi::BaseMPIWorkDistributor](classdynampi_1_1BaseMPIWorkDistributor.md)












## Classes

| Type | Name |
| ---: | :--- |
| struct | [**Config**](structdynampi_1_1HierarchicalMPIWorkDistributor_1_1Config.md) <br> |
| struct | [**RunConfig**](structdynampi_1_1HierarchicalMPIWorkDistributor_1_1RunConfig.md) <br> |














## Public Static Attributes

| Type | Name |
| ---: | :--- |
|  [**const**](structdynampi_1_1MPI__Type.md) [**bool**](structdynampi_1_1MPI__Type.md) | [**ordered**](#variable-ordered)   = `[**false**](structdynampi_1_1MPI__Type.md)`<br> |
|  [**constexpr**](structdynampi_1_1MPI__Type.md) [**bool**](structdynampi_1_1MPI__Type.md) | [**prioritize\_tasks**](#variable-prioritize_tasks)   = `Base::prioritize\_tasks`<br> |




























## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**HierarchicalMPIWorkDistributor**](#function-hierarchicalmpiworkdistributor) (std::function&lt; [**ResultT**](structdynampi_1_1MPI__Type.md)([**TaskT**](structdynampi_1_1MPI__Type.md))&gt; worker\_function, [**Config**](structdynampi_1_1HierarchicalMPIWorkDistributor_1_1Config.md) runtime\_config=[**Config**](structdynampi_1_1HierarchicalMPIWorkDistributor_1_1Config.md){}) <br> |
|  [**void**](structdynampi_1_1MPI__Type.md) | [**allocate\_task\_to\_child**](#function-allocate_task_to_child) () <br> |
|  [**void**](structdynampi_1_1MPI__Type.md) | [**finalize**](#function-finalize) () <br> |
|  std::vector&lt; [**ResultT**](structdynampi_1_1MPI__Type.md) &gt; | [**finish\_remaining\_tasks**](#function-finish_remaining_tasks) () <br> |
|  [**const**](structdynampi_1_1MPI__Type.md) StatisticsT & | [**get\_statistics**](#function-get_statistics) () const<br> |
|  [**void**](structdynampi_1_1MPI__Type.md) | [**insert\_task**](#function-insert_task-12) ([**TaskT**](structdynampi_1_1MPI__Type.md) task) <br> |
|  [**void**](structdynampi_1_1MPI__Type.md) | [**insert\_task**](#function-insert_task-22) ([**const**](structdynampi_1_1MPI__Type.md) [**TaskT**](structdynampi_1_1MPI__Type.md) & task, [**double**](structdynampi_1_1MPI__Type.md) priority) <br> |
|  [**void**](structdynampi_1_1MPI__Type.md) | [**insert\_tasks**](#function-insert_tasks-12) ([**const**](structdynampi_1_1MPI__Type.md) [**Range**](structdynampi_1_1MPI__Type.md) & tasks) <br> |
|  [**void**](structdynampi_1_1MPI__Type.md) | [**insert\_tasks**](#function-insert_tasks-22) ([**const**](structdynampi_1_1MPI__Type.md) std::vector&lt; [**TaskT**](structdynampi_1_1MPI__Type.md) &gt; & tasks) <br> |
|  [**bool**](structdynampi_1_1MPI__Type.md) | [**is\_root\_manager**](#function-is_root_manager) () const<br> |
|  [**size\_t**](structdynampi_1_1MPI__Type.md) | [**remaining\_tasks\_count**](#function-remaining_tasks_count) () const<br> |
|  [**void**](structdynampi_1_1MPI__Type.md) | [**return\_results\_and\_request\_next\_batch\_from\_manager**](#function-return_results_and_request_next_batch_from_manager) () <br> |
|  std::vector&lt; [**ResultT**](structdynampi_1_1MPI__Type.md) &gt; | [**run\_tasks**](#function-run_tasks) ([**const**](structdynampi_1_1MPI__Type.md) [**RunConfig**](structdynampi_1_1HierarchicalMPIWorkDistributor_1_1RunConfig.md) & config=[**RunConfig**](structdynampi_1_1HierarchicalMPIWorkDistributor_1_1RunConfig.md){}) <br> |
|  [**void**](structdynampi_1_1MPI__Type.md) | [**run\_worker**](#function-run_worker) () <br> |
|   | [**~HierarchicalMPIWorkDistributor**](#function-hierarchicalmpiworkdistributor) () <br> |










## Protected Types inherited from dynampi::BaseMPIWorkDistributor

See [dynampi::BaseMPIWorkDistributor](classdynampi_1_1BaseMPIWorkDistributor.md)

| Type | Name |
| ---: | :--- |
| typedef std::conditional\_t&lt; [**prioritize\_tasks**](classdynampi_1_1BaseMPIWorkDistributor.md#variable-prioritize_tasks), std::priority\_queue&lt; std::pair&lt; [**double**](structdynampi_1_1MPI__Type.md), [**TaskT**](structdynampi_1_1MPI__Type.md) &gt; &gt;, std::deque&lt; [**TaskT**](structdynampi_1_1MPI__Type.md) &gt; &gt; | [**QueueT**](classdynampi_1_1BaseMPIWorkDistributor.md#typedef-queuet)  <br> |












## Protected Static Attributes inherited from dynampi::BaseMPIWorkDistributor

See [dynampi::BaseMPIWorkDistributor](classdynampi_1_1BaseMPIWorkDistributor.md)

| Type | Name |
| ---: | :--- |
|  [**constexpr**](structdynampi_1_1MPI__Type.md) [**bool**](structdynampi_1_1MPI__Type.md) | [**prioritize\_tasks**](classdynampi_1_1BaseMPIWorkDistributor.md#variable-prioritize_tasks)   = `[**get\_option\_value**](template__options_8hpp.md#function-get_option_value)&lt;[**prioritize\_tasks\_t**](structdynampi_1_1prioritize__tasks__t.md), Options...&gt;()`<br> |


































## Public Static Attributes Documentation




### variable ordered 

```C++
const bool dynampi::HierarchicalMPIWorkDistributor< TaskT, ResultT, Options >::ordered;
```




<hr>



### variable prioritize\_tasks 

```C++
constexpr bool dynampi::HierarchicalMPIWorkDistributor< TaskT, ResultT, Options >::prioritize_tasks;
```




<hr>
## Public Functions Documentation




### function HierarchicalMPIWorkDistributor 

```C++
inline explicit dynampi::HierarchicalMPIWorkDistributor::HierarchicalMPIWorkDistributor (
    std::function< ResultT ( TaskT )> worker_function,
    Config runtime_config=Config {}
) 
```




<hr>



### function allocate\_task\_to\_child 

```C++
inline void dynampi::HierarchicalMPIWorkDistributor::allocate_task_to_child () 
```




<hr>



### function finalize 

```C++
inline void dynampi::HierarchicalMPIWorkDistributor::finalize () 
```




<hr>



### function finish\_remaining\_tasks 

```C++
inline std::vector< ResultT > dynampi::HierarchicalMPIWorkDistributor::finish_remaining_tasks () 
```




<hr>



### function get\_statistics 

```C++
inline const StatisticsT & dynampi::HierarchicalMPIWorkDistributor::get_statistics () const
```




<hr>



### function insert\_task [1/2]

```C++
inline void dynampi::HierarchicalMPIWorkDistributor::insert_task (
    TaskT task
) 
```




<hr>



### function insert\_task [2/2]

```C++
inline void dynampi::HierarchicalMPIWorkDistributor::insert_task (
    const  TaskT & task,
    double priority
) 
```




<hr>



### function insert\_tasks [1/2]

```C++
template<typename  Range>
inline void dynampi::HierarchicalMPIWorkDistributor::insert_tasks (
    const  Range & tasks
) 
```




<hr>



### function insert\_tasks [2/2]

```C++
inline void dynampi::HierarchicalMPIWorkDistributor::insert_tasks (
    const std::vector< TaskT > & tasks
) 
```




<hr>



### function is\_root\_manager 

```C++
inline bool dynampi::HierarchicalMPIWorkDistributor::is_root_manager () const
```




<hr>



### function remaining\_tasks\_count 

```C++
inline size_t dynampi::HierarchicalMPIWorkDistributor::remaining_tasks_count () const
```




<hr>



### function return\_results\_and\_request\_next\_batch\_from\_manager 

```C++
inline void dynampi::HierarchicalMPIWorkDistributor::return_results_and_request_next_batch_from_manager () 
```




<hr>



### function run\_tasks 

```C++
inline std::vector< ResultT > dynampi::HierarchicalMPIWorkDistributor::run_tasks (
    const  RunConfig & config=RunConfig {}
) 
```




<hr>



### function run\_worker 

```C++
inline void dynampi::HierarchicalMPIWorkDistributor::run_worker () 
```




<hr>



### function ~HierarchicalMPIWorkDistributor 

```C++
inline dynampi::HierarchicalMPIWorkDistributor::~HierarchicalMPIWorkDistributor () 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/hierarchical_distributor.hpp`

