

# Class dynampi::LockFreeRMAWorkDistributor

**template &lt;[**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type), typename... Options&gt;**



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**LockFreeRMAWorkDistributor**](classdynampi_1_1LockFreeRMAWorkDistributor.md)





* `#include <lockfree_rma_distributor.hpp>`















## Classes

| Type | Name |
| ---: | :--- |
| struct | [**Config**](structdynampi_1_1LockFreeRMAWorkDistributor_1_1Config.md) <br> |
| struct | [**RunConfig**](structdynampi_1_1LockFreeRMAWorkDistributor_1_1RunConfig.md) <br> |
| struct | [**Statistics**](structdynampi_1_1LockFreeRMAWorkDistributor_1_1Statistics.md) <br> |


## Public Types

| Type | Name |
| ---: | :--- |
| typedef std::conditional\_t&lt; statistics\_mode !=StatisticsMode::None, [**Statistics**](structdynampi_1_1LockFreeRMAWorkDistributor_1_1Statistics.md), std::monostate &gt; | [**StatisticsT**](#typedef-statisticst)  <br> |






## Public Static Attributes

| Type | Name |
| ---: | :--- |
|  [**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**ordered**](#variable-ordered)   = `[**true**](namespacedynampi.md#function-check_fixed_size_mpi_type)`<br> |














## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**LockFreeRMAWorkDistributor**](#function-lockfreermaworkdistributor) (std::function&lt; [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type)([**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type))&gt; worker\_function, [**Config**](structdynampi_1_1LockFreeRMAWorkDistributor_1_1Config.md) config={}) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**finalize**](#function-finalize) () <br> |
|  std::vector&lt; [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; | [**finish\_remaining\_tasks**](#function-finish_remaining_tasks) () <br> |
|  std::vector&lt; [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; | [**gather\_once**](#function-gather_once) () <br> |
|  [**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**StatisticsT**](classdynampi_1_1LockFreeRMAWorkDistributor.md#typedef-statisticst) & | [**get\_statistics**](#function-get_statistics) () const<br> |
|  [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**has\_task\_errors**](#function-has_task_errors) () const<br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**insert\_task**](#function-insert_task-12) ([**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type) task) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**insert\_task**](#function-insert_task-22) ([**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type) & task, [**double**](namespacedynampi.md#function-check_fixed_size_mpi_type)) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**insert\_tasks**](#function-insert_tasks) ([**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) std::vector&lt; [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; & tasks) <br> |
|  [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**is\_root\_manager**](#function-is_root_manager) () const<br> |
|  [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**num\_workers**](#function-num_workers) () const<br> |
|  [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**remaining\_tasks\_count**](#function-remaining_tasks_count) () const<br> |
|  std::vector&lt; [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; | [**run\_tasks**](#function-run_tasks) ([**RunConfig**](structdynampi_1_1LockFreeRMAWorkDistributor_1_1RunConfig.md) config={}) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**run\_worker**](#function-run_worker) () <br> |
|  std::vector&lt; [**TaskError**](structdynampi_1_1TaskError.md) &gt; | [**take\_task\_errors**](#function-take_task_errors) () <br> |
|   | [**~LockFreeRMAWorkDistributor**](#function-lockfreermaworkdistributor) () <br> |




























## Public Types Documentation




### typedef StatisticsT 

```C++
using dynampi::LockFreeRMAWorkDistributor< TaskT, ResultT, Options >::StatisticsT =  std::conditional_t<statistics_mode != StatisticsMode::None, Statistics, std::monostate>;
```




<hr>
## Public Static Attributes Documentation




### variable ordered 

```C++
const bool dynampi::LockFreeRMAWorkDistributor< TaskT, ResultT, Options >::ordered;
```




<hr>
## Public Functions Documentation




### function LockFreeRMAWorkDistributor 

```C++
inline explicit dynampi::LockFreeRMAWorkDistributor::LockFreeRMAWorkDistributor (
    std::function< ResultT ( TaskT )> worker_function,
    Config config={}
) 
```




<hr>



### function finalize 

```C++
inline void dynampi::LockFreeRMAWorkDistributor::finalize () 
```




<hr>



### function finish\_remaining\_tasks 

```C++
inline std::vector< ResultT > dynampi::LockFreeRMAWorkDistributor::finish_remaining_tasks () 
```




<hr>



### function gather\_once 

```C++
inline std::vector< ResultT > dynampi::LockFreeRMAWorkDistributor::gather_once () 
```




<hr>



### function get\_statistics 

```C++
inline const  StatisticsT & dynampi::LockFreeRMAWorkDistributor::get_statistics () const
```




<hr>



### function has\_task\_errors 

```C++
inline bool dynampi::LockFreeRMAWorkDistributor::has_task_errors () const
```




<hr>



### function insert\_task [1/2]

```C++
inline void dynampi::LockFreeRMAWorkDistributor::insert_task (
    TaskT task
) 
```




<hr>



### function insert\_task [2/2]

```C++
inline void dynampi::LockFreeRMAWorkDistributor::insert_task (
    const  TaskT & task,
    double
) 
```




<hr>



### function insert\_tasks 

```C++
inline void dynampi::LockFreeRMAWorkDistributor::insert_tasks (
    const std::vector< TaskT > & tasks
) 
```




<hr>



### function is\_root\_manager 

```C++
inline bool dynampi::LockFreeRMAWorkDistributor::is_root_manager () const
```




<hr>



### function num\_workers 

```C++
inline int dynampi::LockFreeRMAWorkDistributor::num_workers () const
```




<hr>



### function remaining\_tasks\_count 

```C++
inline size_t dynampi::LockFreeRMAWorkDistributor::remaining_tasks_count () const
```




<hr>



### function run\_tasks 

```C++
inline std::vector< ResultT > dynampi::LockFreeRMAWorkDistributor::run_tasks (
    RunConfig config={}
) 
```




<hr>



### function run\_worker 

```C++
inline void dynampi::LockFreeRMAWorkDistributor::run_worker () 
```




<hr>



### function take\_task\_errors 

```C++
inline std::vector< TaskError > dynampi::LockFreeRMAWorkDistributor::take_task_errors () 
```




<hr>



### function ~LockFreeRMAWorkDistributor 

```C++
inline dynampi::LockFreeRMAWorkDistributor::~LockFreeRMAWorkDistributor () 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/lockfree_rma_distributor.hpp`

