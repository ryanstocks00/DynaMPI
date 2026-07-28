

# Class dynampi::HierarchicalNonBlockingMPIWorkDistributor

**template &lt;typename TaskT, typename ResultT, typename... Options&gt;**



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**HierarchicalNonBlockingMPIWorkDistributor**](classdynampi_1_1HierarchicalNonBlockingMPIWorkDistributor.md)





* `#include <hierarchical_nonblocking_distributor.hpp>`



Inherits the following classes: [dynampi::BaseMPIWorkDistributor](classdynampi_1_1BaseMPIWorkDistributor.md)












## Classes

| Type | Name |
| ---: | :--- |
| struct | [**Config**](structdynampi_1_1HierarchicalNonBlockingMPIWorkDistributor_1_1Config.md) <br> |
| struct | [**RunConfig**](structdynampi_1_1HierarchicalNonBlockingMPIWorkDistributor_1_1RunConfig.md) <br> |














## Public Static Attributes

| Type | Name |
| ---: | :--- |
|  const bool | [**ordered**](#variable-ordered)   = `false`<br> |
|  constexpr bool | [**prioritize\_tasks**](#variable-prioritize_tasks)   = `Base::prioritize\_tasks`<br> |




























## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**HierarchicalNonBlockingMPIWorkDistributor**](#function-hierarchicalnonblockingmpiworkdistributor) (std::function&lt; ResultT(TaskT)&gt; worker\_function, [**Config**](structdynampi_1_1HierarchicalNonBlockingMPIWorkDistributor_1_1Config.md) runtime\_config=[**Config**](structdynampi_1_1HierarchicalNonBlockingMPIWorkDistributor_1_1Config.md){}) <br> |
|  void | [**allocate\_task\_to\_child**](#function-allocate_task_to_child) () <br> |
|  void | [**finalize**](#function-finalize) () <br> |
|  std::vector&lt; ResultT &gt; | [**finish\_remaining\_tasks**](#function-finish_remaining_tasks) () <br> |
|  const StatisticsT & | [**get\_statistics**](#function-get_statistics) () const<br> |
|  void | [**insert\_task**](#function-insert_task-12) (TaskT task) <br> |
|  void | [**insert\_task**](#function-insert_task-22) (const TaskT & task, double priority) <br> |
|  void | [**insert\_tasks**](#function-insert_tasks-12) (const Range & tasks) <br> |
|  void | [**insert\_tasks**](#function-insert_tasks-22) (const std::vector&lt; TaskT &gt; & tasks) <br> |
|  bool | [**is\_root\_manager**](#function-is_root_manager) () const<br> |
|  size\_t | [**remaining\_tasks\_count**](#function-remaining_tasks_count) () const<br> |
|  std::vector&lt; ResultT &gt; | [**run\_tasks**](#function-run_tasks) (const [**RunConfig**](structdynampi_1_1HierarchicalNonBlockingMPIWorkDistributor_1_1RunConfig.md) & config=[**RunConfig**](structdynampi_1_1HierarchicalNonBlockingMPIWorkDistributor_1_1RunConfig.md){}) <br> |
|  void | [**run\_worker**](#function-run_worker) () <br> |
|  void | [**send\_results\_to\_parent**](#function-send_results_to_parent) () <br> |
|   | [**~HierarchicalNonBlockingMPIWorkDistributor**](#function-hierarchicalnonblockingmpiworkdistributor) () <br> |










## Protected Types inherited from dynampi::BaseMPIWorkDistributor

See [dynampi::BaseMPIWorkDistributor](classdynampi_1_1BaseMPIWorkDistributor.md)

| Type | Name |
| ---: | :--- |
| typedef std::conditional\_t&lt; [**prioritize\_tasks**](classdynampi_1_1BaseMPIWorkDistributor.md#variable-prioritize_tasks), std::priority\_queue&lt; std::pair&lt; double, TaskT &gt; &gt;, std::deque&lt; TaskT &gt; &gt; | [**QueueT**](classdynampi_1_1BaseMPIWorkDistributor.md#typedef-queuet)  <br> |












## Protected Static Attributes inherited from dynampi::BaseMPIWorkDistributor

See [dynampi::BaseMPIWorkDistributor](classdynampi_1_1BaseMPIWorkDistributor.md)

| Type | Name |
| ---: | :--- |
|  constexpr bool | [**prioritize\_tasks**](classdynampi_1_1BaseMPIWorkDistributor.md#variable-prioritize_tasks)   = `[**get\_option\_value**](template__options_8hpp.md#function-get_option_value)&lt;[**prioritize\_tasks\_t**](structdynampi_1_1prioritize__tasks__t.md), Options...&gt;()`<br> |


































## Public Static Attributes Documentation




### variable ordered 

```C++
const bool dynampi::HierarchicalNonBlockingMPIWorkDistributor< TaskT, ResultT, Options >::ordered;
```




<hr>



### variable prioritize\_tasks 

```C++
constexpr bool dynampi::HierarchicalNonBlockingMPIWorkDistributor< TaskT, ResultT, Options >::prioritize_tasks;
```




<hr>
## Public Functions Documentation




### function HierarchicalNonBlockingMPIWorkDistributor 

```C++
inline explicit dynampi::HierarchicalNonBlockingMPIWorkDistributor::HierarchicalNonBlockingMPIWorkDistributor (
    std::function< ResultT(TaskT)> worker_function,
    Config runtime_config=Config {}
) 
```




<hr>



### function allocate\_task\_to\_child 

```C++
inline void dynampi::HierarchicalNonBlockingMPIWorkDistributor::allocate_task_to_child () 
```




<hr>



### function finalize 

```C++
inline void dynampi::HierarchicalNonBlockingMPIWorkDistributor::finalize () 
```




<hr>



### function finish\_remaining\_tasks 

```C++
inline std::vector< ResultT > dynampi::HierarchicalNonBlockingMPIWorkDistributor::finish_remaining_tasks () 
```




<hr>



### function get\_statistics 

```C++
inline const StatisticsT & dynampi::HierarchicalNonBlockingMPIWorkDistributor::get_statistics () const
```




<hr>



### function insert\_task [1/2]

```C++
inline void dynampi::HierarchicalNonBlockingMPIWorkDistributor::insert_task (
    TaskT task
) 
```




<hr>



### function insert\_task [2/2]

```C++
inline void dynampi::HierarchicalNonBlockingMPIWorkDistributor::insert_task (
    const TaskT & task,
    double priority
) 
```




<hr>



### function insert\_tasks [1/2]

```C++
template<typename Range>
inline void dynampi::HierarchicalNonBlockingMPIWorkDistributor::insert_tasks (
    const Range & tasks
) 
```




<hr>



### function insert\_tasks [2/2]

```C++
inline void dynampi::HierarchicalNonBlockingMPIWorkDistributor::insert_tasks (
    const std::vector< TaskT > & tasks
) 
```




<hr>



### function is\_root\_manager 

```C++
inline bool dynampi::HierarchicalNonBlockingMPIWorkDistributor::is_root_manager () const
```




<hr>



### function remaining\_tasks\_count 

```C++
inline size_t dynampi::HierarchicalNonBlockingMPIWorkDistributor::remaining_tasks_count () const
```




<hr>



### function run\_tasks 

```C++
inline std::vector< ResultT > dynampi::HierarchicalNonBlockingMPIWorkDistributor::run_tasks (
    const RunConfig & config=RunConfig {}
) 
```




<hr>



### function run\_worker 

```C++
inline void dynampi::HierarchicalNonBlockingMPIWorkDistributor::run_worker () 
```




<hr>



### function send\_results\_to\_parent 

```C++
inline void dynampi::HierarchicalNonBlockingMPIWorkDistributor::send_results_to_parent () 
```




<hr>



### function ~HierarchicalNonBlockingMPIWorkDistributor 

```C++
inline dynampi::HierarchicalNonBlockingMPIWorkDistributor::~HierarchicalNonBlockingMPIWorkDistributor () 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/hierarchical_nonblocking_distributor.hpp`

