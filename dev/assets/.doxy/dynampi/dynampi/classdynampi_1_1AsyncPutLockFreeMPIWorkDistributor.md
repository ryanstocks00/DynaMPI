

# Class dynampi::AsyncPutLockFreeMPIWorkDistributor

**template &lt;typename TaskT, typename ResultT, typename... Options&gt;**



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**AsyncPutLockFreeMPIWorkDistributor**](classdynampi_1_1AsyncPutLockFreeMPIWorkDistributor.md)





* `#include <async_put_lockfree_distributor.hpp>`















## Classes

| Type | Name |
| ---: | :--- |
| struct | [**Config**](structdynampi_1_1AsyncPutLockFreeMPIWorkDistributor_1_1Config.md) <br> |
| struct | [**RunConfig**](structdynampi_1_1AsyncPutLockFreeMPIWorkDistributor_1_1RunConfig.md) <br> |
| struct | [**Statistics**](structdynampi_1_1AsyncPutLockFreeMPIWorkDistributor_1_1Statistics.md) <br> |


## Public Types

| Type | Name |
| ---: | :--- |
| typedef std::conditional\_t&lt; statistics\_mode !=StatisticsMode::None, [**Statistics**](structdynampi_1_1AsyncPutLockFreeMPIWorkDistributor_1_1Statistics.md), std::monostate &gt; | [**StatisticsT**](#typedef-statisticst)  <br> |






## Public Static Attributes

| Type | Name |
| ---: | :--- |
|  const bool | [**ordered**](#variable-ordered)   = `false`<br> |














## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**AsyncPutLockFreeMPIWorkDistributor**](#function-asyncputlockfreempiworkdistributor) (std::function&lt; ResultT(TaskT)&gt; worker\_function, [**Config**](structdynampi_1_1AsyncPutLockFreeMPIWorkDistributor_1_1Config.md) config={}) <br> |
|  void | [**finalize**](#function-finalize) () <br> |
|  std::vector&lt; ResultT &gt; | [**finish\_remaining\_tasks**](#function-finish_remaining_tasks) () <br> |
|  std::vector&lt; ResultT &gt; | [**gather\_once**](#function-gather_once) () <br> |
|  const [**StatisticsT**](classdynampi_1_1AsyncPutLockFreeMPIWorkDistributor.md#typedef-statisticst) & | [**get\_statistics**](#function-get_statistics) () const<br> |
|  void | [**insert\_task**](#function-insert_task-12) (TaskT task) <br> |
|  void | [**insert\_task**](#function-insert_task-22) (const TaskT & task, double) <br> |
|  void | [**insert\_tasks**](#function-insert_tasks) (const std::vector&lt; TaskT &gt; & tasks) <br> |
|  bool | [**is\_root\_manager**](#function-is_root_manager) () const<br> |
|  int | [**num\_workers**](#function-num_workers) () const<br> |
|  size\_t | [**remaining\_tasks\_count**](#function-remaining_tasks_count) () const<br> |
|  std::vector&lt; ResultT &gt; | [**run\_tasks**](#function-run_tasks) ([**RunConfig**](structdynampi_1_1AsyncPutLockFreeMPIWorkDistributor_1_1RunConfig.md) config={}) <br> |
|  void | [**run\_worker**](#function-run_worker) () <br> |
|   | [**~AsyncPutLockFreeMPIWorkDistributor**](#function-asyncputlockfreempiworkdistributor) () <br> |




























## Public Types Documentation




### typedef StatisticsT 

```C++
using dynampi::AsyncPutLockFreeMPIWorkDistributor< TaskT, ResultT, Options >::StatisticsT =  std::conditional_t<statistics_mode != StatisticsMode::None, Statistics, std::monostate>;
```




<hr>
## Public Static Attributes Documentation




### variable ordered 

```C++
const bool dynampi::AsyncPutLockFreeMPIWorkDistributor< TaskT, ResultT, Options >::ordered;
```




<hr>
## Public Functions Documentation




### function AsyncPutLockFreeMPIWorkDistributor 

```C++
inline explicit dynampi::AsyncPutLockFreeMPIWorkDistributor::AsyncPutLockFreeMPIWorkDistributor (
    std::function< ResultT(TaskT)> worker_function,
    Config config={}
) 
```




<hr>



### function finalize 

```C++
inline void dynampi::AsyncPutLockFreeMPIWorkDistributor::finalize () 
```




<hr>



### function finish\_remaining\_tasks 

```C++
inline std::vector< ResultT > dynampi::AsyncPutLockFreeMPIWorkDistributor::finish_remaining_tasks () 
```




<hr>



### function gather\_once 

```C++
inline std::vector< ResultT > dynampi::AsyncPutLockFreeMPIWorkDistributor::gather_once () 
```




<hr>



### function get\_statistics 

```C++
inline const StatisticsT & dynampi::AsyncPutLockFreeMPIWorkDistributor::get_statistics () const
```




<hr>



### function insert\_task [1/2]

```C++
inline void dynampi::AsyncPutLockFreeMPIWorkDistributor::insert_task (
    TaskT task
) 
```




<hr>



### function insert\_task [2/2]

```C++
inline void dynampi::AsyncPutLockFreeMPIWorkDistributor::insert_task (
    const TaskT & task,
    double
) 
```




<hr>



### function insert\_tasks 

```C++
inline void dynampi::AsyncPutLockFreeMPIWorkDistributor::insert_tasks (
    const std::vector< TaskT > & tasks
) 
```




<hr>



### function is\_root\_manager 

```C++
inline bool dynampi::AsyncPutLockFreeMPIWorkDistributor::is_root_manager () const
```




<hr>



### function num\_workers 

```C++
inline int dynampi::AsyncPutLockFreeMPIWorkDistributor::num_workers () const
```




<hr>



### function remaining\_tasks\_count 

```C++
inline size_t dynampi::AsyncPutLockFreeMPIWorkDistributor::remaining_tasks_count () const
```




<hr>



### function run\_tasks 

```C++
inline std::vector< ResultT > dynampi::AsyncPutLockFreeMPIWorkDistributor::run_tasks (
    RunConfig config={}
) 
```




<hr>



### function run\_worker 

```C++
inline void dynampi::AsyncPutLockFreeMPIWorkDistributor::run_worker () 
```




<hr>



### function ~AsyncPutLockFreeMPIWorkDistributor 

```C++
inline dynampi::AsyncPutLockFreeMPIWorkDistributor::~AsyncPutLockFreeMPIWorkDistributor () 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/async_put_lockfree_distributor.hpp`

