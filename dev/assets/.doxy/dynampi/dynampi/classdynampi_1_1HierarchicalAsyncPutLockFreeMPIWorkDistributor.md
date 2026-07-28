

# Class dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor

**template &lt;typename TaskT, typename ResultT, typename... Options&gt;**



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**HierarchicalAsyncPutLockFreeMPIWorkDistributor**](classdynampi_1_1HierarchicalAsyncPutLockFreeMPIWorkDistributor.md)





* `#include <hierarchical_async_put_lockfree_distributor.hpp>`















## Classes

| Type | Name |
| ---: | :--- |
| struct | [**Config**](structdynampi_1_1HierarchicalAsyncPutLockFreeMPIWorkDistributor_1_1Config.md) <br> |
| struct | [**RunConfig**](structdynampi_1_1HierarchicalAsyncPutLockFreeMPIWorkDistributor_1_1RunConfig.md) <br> |








## Public Static Attributes

| Type | Name |
| ---: | :--- |
|  const bool | [**ordered**](#variable-ordered)   = `false`<br> |














## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**HierarchicalAsyncPutLockFreeMPIWorkDistributor**](#function-hierarchicalasyncputlockfreempiworkdistributor) (std::function&lt; ResultT(TaskT)&gt; worker\_function, [**Config**](structdynampi_1_1HierarchicalAsyncPutLockFreeMPIWorkDistributor_1_1Config.md) config={}) <br> |
|  void | [**finalize**](#function-finalize) () <br> |
|  std::vector&lt; ResultT &gt; | [**finish\_remaining\_tasks**](#function-finish_remaining_tasks) () <br> |
|  std::vector&lt; ResultT &gt; | [**gather\_once**](#function-gather_once) () <br> |
|  void | [**insert\_task**](#function-insert_task) (TaskT task) <br> |
|  void | [**insert\_tasks**](#function-insert_tasks) (const std::vector&lt; TaskT &gt; & tasks) <br> |
|  bool | [**is\_root\_manager**](#function-is_root_manager) () const<br> |
|  size\_t | [**remaining\_tasks\_count**](#function-remaining_tasks_count) () const<br> |
|  std::vector&lt; ResultT &gt; | [**run\_tasks**](#function-run_tasks) (const [**RunConfig**](structdynampi_1_1HierarchicalAsyncPutLockFreeMPIWorkDistributor_1_1RunConfig.md) & config=[**RunConfig**](structdynampi_1_1HierarchicalAsyncPutLockFreeMPIWorkDistributor_1_1RunConfig.md){}) <br> |
|  void | [**run\_worker**](#function-run_worker) () <br> |
|   | [**~HierarchicalAsyncPutLockFreeMPIWorkDistributor**](#function-hierarchicalasyncputlockfreempiworkdistributor) () <br> |




























## Public Static Attributes Documentation




### variable ordered 

```C++
const bool dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor< TaskT, ResultT, Options >::ordered;
```




<hr>
## Public Functions Documentation




### function HierarchicalAsyncPutLockFreeMPIWorkDistributor 

```C++
inline explicit dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor::HierarchicalAsyncPutLockFreeMPIWorkDistributor (
    std::function< ResultT(TaskT)> worker_function,
    Config config={}
) 
```




<hr>



### function finalize 

```C++
inline void dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor::finalize () 
```




<hr>



### function finish\_remaining\_tasks 

```C++
inline std::vector< ResultT > dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor::finish_remaining_tasks () 
```




<hr>



### function gather\_once 

```C++
inline std::vector< ResultT > dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor::gather_once () 
```




<hr>



### function insert\_task 

```C++
inline void dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor::insert_task (
    TaskT task
) 
```




<hr>



### function insert\_tasks 

```C++
inline void dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor::insert_tasks (
    const std::vector< TaskT > & tasks
) 
```




<hr>



### function is\_root\_manager 

```C++
inline bool dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor::is_root_manager () const
```




<hr>



### function remaining\_tasks\_count 

```C++
inline size_t dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor::remaining_tasks_count () const
```




<hr>



### function run\_tasks 

```C++
inline std::vector< ResultT > dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor::run_tasks (
    const RunConfig & config=RunConfig {}
) 
```




<hr>



### function run\_worker 

```C++
inline void dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor::run_worker () 
```




<hr>



### function ~HierarchicalAsyncPutLockFreeMPIWorkDistributor 

```C++
inline dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor::~HierarchicalAsyncPutLockFreeMPIWorkDistributor () 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/hierarchical_async_put_lockfree_distributor.hpp`

