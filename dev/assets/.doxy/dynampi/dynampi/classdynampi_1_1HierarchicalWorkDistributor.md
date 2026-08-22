

# Class dynampi::HierarchicalWorkDistributor

**template &lt;[**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type), typename... Options&gt;**



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**HierarchicalWorkDistributor**](classdynampi_1_1HierarchicalWorkDistributor.md)





* `#include <hierarchical_distributor.hpp>`



Inherits the following classes: [dynampi::BaseWorkDistributor](classdynampi_1_1BaseWorkDistributor.md)












## Classes

| Type | Name |
| ---: | :--- |
| struct | [**Config**](structdynampi_1_1HierarchicalWorkDistributor_1_1Config.md) <br> |
| struct | [**RunConfig**](structdynampi_1_1HierarchicalWorkDistributor_1_1RunConfig.md) <br> |














## Public Static Attributes

| Type | Name |
| ---: | :--- |
|  [**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**ordered**](#variable-ordered)   = `[**false**](namespacedynampi.md#function-check_fixed_size_mpi_type)`<br> |
|  [**constexpr**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**prioritize\_tasks**](#variable-prioritize_tasks)   = `Base::prioritize\_tasks`<br> |




























## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**HierarchicalWorkDistributor**](#function-hierarchicalworkdistributor) (std::function&lt; [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type)([**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type))&gt; worker\_function, [**Config**](structdynampi_1_1HierarchicalWorkDistributor_1_1Config.md) runtime\_config=[**Config**](structdynampi_1_1HierarchicalWorkDistributor_1_1Config.md){}) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**allocate\_task\_to\_child**](#function-allocate_task_to_child) () <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**finalize**](#function-finalize) () <br> |
|  std::vector&lt; [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; | [**finish\_remaining\_tasks**](#function-finish_remaining_tasks) () <br> |
|  [**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) StatisticsT & | [**get\_statistics**](#function-get_statistics) () const<br> |
|  [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**has\_task\_errors**](#function-has_task_errors) () const<br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**insert\_task**](#function-insert_task-12) ([**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type) task) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**insert\_task**](#function-insert_task-22) ([**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type) & task, [**double**](namespacedynampi.md#function-check_fixed_size_mpi_type) priority) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**insert\_tasks**](#function-insert_tasks-12) ([**Range**](namespacedynampi.md#function-check_fixed_size_mpi_type) && tasks) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**insert\_tasks**](#function-insert_tasks-22) ([**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) std::vector&lt; [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; & tasks) <br> |
|  [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**is\_root\_manager**](#function-is_root_manager) () const<br> |
|  [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**remaining\_tasks\_count**](#function-remaining_tasks_count) () const<br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**request\_next\_batch\_if\_room**](#function-request_next_batch_if_room) () <br> |
|  std::vector&lt; [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; | [**run\_tasks**](#function-run_tasks) ([**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**RunConfig**](structdynampi_1_1HierarchicalWorkDistributor_1_1RunConfig.md) & config=[**RunConfig**](structdynampi_1_1HierarchicalWorkDistributor_1_1RunConfig.md){}) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**run\_worker**](#function-run_worker) () <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**send\_results\_to\_parent**](#function-send_results_to_parent) () <br> |
|  std::vector&lt; [**TaskError**](structdynampi_1_1TaskError.md) &gt; | [**take\_task\_errors**](#function-take_task_errors) () <br> |
|   | [**~HierarchicalWorkDistributor**](#function-hierarchicalworkdistributor) () <br> |










## Protected Types inherited from dynampi::BaseWorkDistributor

See [dynampi::BaseWorkDistributor](classdynampi_1_1BaseWorkDistributor.md)

| Type | Name |
| ---: | :--- |
| typedef std::conditional\_t&lt; [**prioritize\_tasks**](classdynampi_1_1BaseWorkDistributor.md#variable-prioritize_tasks), std::priority\_queue&lt; std::pair&lt; [**double**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; &gt;, std::deque&lt; [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; &gt; | [**QueueT**](classdynampi_1_1BaseWorkDistributor.md#typedef-queuet)  <br> |












## Protected Static Attributes inherited from dynampi::BaseWorkDistributor

See [dynampi::BaseWorkDistributor](classdynampi_1_1BaseWorkDistributor.md)

| Type | Name |
| ---: | :--- |
|  [**constexpr**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**prioritize\_tasks**](classdynampi_1_1BaseWorkDistributor.md#variable-prioritize_tasks)   = `[**get\_option\_value**](template__options_8hpp.md#function-get_option_value)&lt;[**prioritize\_tasks\_t**](structdynampi_1_1prioritize__tasks__t.md), Options...&gt;()`<br> |


































## Public Static Attributes Documentation




### variable ordered 

```C++
const bool dynampi::HierarchicalWorkDistributor< TaskT, ResultT, Options >::ordered;
```




<hr>



### variable prioritize\_tasks 

```C++
constexpr bool dynampi::HierarchicalWorkDistributor< TaskT, ResultT, Options >::prioritize_tasks;
```




<hr>
## Public Functions Documentation




### function HierarchicalWorkDistributor 

```C++
inline explicit dynampi::HierarchicalWorkDistributor::HierarchicalWorkDistributor (
    std::function< ResultT ( TaskT )> worker_function,
    Config runtime_config=Config {}
) 
```




<hr>



### function allocate\_task\_to\_child 

```C++
inline void dynampi::HierarchicalWorkDistributor::allocate_task_to_child () 
```




<hr>



### function finalize 

```C++
inline void dynampi::HierarchicalWorkDistributor::finalize () 
```




<hr>



### function finish\_remaining\_tasks 

```C++
inline std::vector< ResultT > dynampi::HierarchicalWorkDistributor::finish_remaining_tasks () 
```




<hr>



### function get\_statistics 

```C++
inline const StatisticsT & dynampi::HierarchicalWorkDistributor::get_statistics () const
```




<hr>



### function has\_task\_errors 

```C++
inline bool dynampi::HierarchicalWorkDistributor::has_task_errors () const
```




<hr>



### function insert\_task [1/2]

```C++
inline void dynampi::HierarchicalWorkDistributor::insert_task (
    TaskT task
) 
```




<hr>



### function insert\_task [2/2]

```C++
inline void dynampi::HierarchicalWorkDistributor::insert_task (
    const  TaskT & task,
    double priority
) 
```




<hr>



### function insert\_tasks [1/2]

```C++
template<typename  Range>
inline void dynampi::HierarchicalWorkDistributor::insert_tasks (
    Range && tasks
) 
```




<hr>



### function insert\_tasks [2/2]

```C++
inline void dynampi::HierarchicalWorkDistributor::insert_tasks (
    const std::vector< TaskT > & tasks
) 
```




<hr>



### function is\_root\_manager 

```C++
inline bool dynampi::HierarchicalWorkDistributor::is_root_manager () const
```




<hr>



### function remaining\_tasks\_count 

```C++
inline size_t dynampi::HierarchicalWorkDistributor::remaining_tasks_count () const
```




<hr>



### function request\_next\_batch\_if\_room 

```C++
inline void dynampi::HierarchicalWorkDistributor::request_next_batch_if_room () 
```




<hr>



### function run\_tasks 

```C++
inline std::vector< ResultT > dynampi::HierarchicalWorkDistributor::run_tasks (
    const  RunConfig & config=RunConfig {}
) 
```




<hr>



### function run\_worker 

```C++
inline void dynampi::HierarchicalWorkDistributor::run_worker () 
```




<hr>



### function send\_results\_to\_parent 

```C++
inline void dynampi::HierarchicalWorkDistributor::send_results_to_parent () 
```




<hr>



### function take\_task\_errors 

```C++
inline std::vector< TaskError > dynampi::HierarchicalWorkDistributor::take_task_errors () 
```




<hr>



### function ~HierarchicalWorkDistributor 

```C++
inline dynampi::HierarchicalWorkDistributor::~HierarchicalWorkDistributor () 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/hierarchical_distributor.hpp`

