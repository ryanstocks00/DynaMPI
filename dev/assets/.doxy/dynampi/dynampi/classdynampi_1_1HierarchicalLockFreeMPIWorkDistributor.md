

# Class dynampi::HierarchicalLockFreeMPIWorkDistributor

**template &lt;typename TaskT, typename ResultT, typename... Options&gt;**



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**HierarchicalLockFreeMPIWorkDistributor**](classdynampi_1_1HierarchicalLockFreeMPIWorkDistributor.md)





* `#include <hierarchical_lockfree_distributor.hpp>`















## Classes

| Type | Name |
| ---: | :--- |
| struct | [**Config**](structdynampi_1_1HierarchicalLockFreeMPIWorkDistributor_1_1Config.md) <br> |
| struct | [**RunConfig**](structdynampi_1_1HierarchicalLockFreeMPIWorkDistributor_1_1RunConfig.md) <br> |








## Public Static Attributes

| Type | Name |
| ---: | :--- |
|  const bool | [**ordered**](#variable-ordered)   = `false`<br> |














## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**HierarchicalLockFreeMPIWorkDistributor**](#function-hierarchicallockfreempiworkdistributor) (std::function&lt; ResultT(TaskT)&gt; worker\_function, [**Config**](structdynampi_1_1HierarchicalLockFreeMPIWorkDistributor_1_1Config.md) config={}) <br> |
|  void | [**finalize**](#function-finalize) () <br> |
|  std::vector&lt; ResultT &gt; | [**finish\_remaining\_tasks**](#function-finish_remaining_tasks) () <br> |
|  void | [**insert\_task**](#function-insert_task) (TaskT task) <br> |
|  void | [**insert\_tasks**](#function-insert_tasks) (const std::vector&lt; TaskT &gt; & tasks) <br> |
|  bool | [**is\_root\_manager**](#function-is_root_manager) () const<br> |
|  size\_t | [**remaining\_tasks\_count**](#function-remaining_tasks_count) () const<br> |
|  std::vector&lt; ResultT &gt; | [**run\_tasks**](#function-run_tasks) (const [**RunConfig**](structdynampi_1_1HierarchicalLockFreeMPIWorkDistributor_1_1RunConfig.md) & config=[**RunConfig**](structdynampi_1_1HierarchicalLockFreeMPIWorkDistributor_1_1RunConfig.md){}) <br> |
|  void | [**run\_worker**](#function-run_worker) () <br> |
|   | [**~HierarchicalLockFreeMPIWorkDistributor**](#function-hierarchicallockfreempiworkdistributor) () <br> |




























## Public Static Attributes Documentation




### variable ordered 

```C++
const bool dynampi::HierarchicalLockFreeMPIWorkDistributor< TaskT, ResultT, Options >::ordered;
```




<hr>
## Public Functions Documentation




### function HierarchicalLockFreeMPIWorkDistributor 

```C++
inline explicit dynampi::HierarchicalLockFreeMPIWorkDistributor::HierarchicalLockFreeMPIWorkDistributor (
    std::function< ResultT(TaskT)> worker_function,
    Config config={}
) 
```




<hr>



### function finalize 

```C++
inline void dynampi::HierarchicalLockFreeMPIWorkDistributor::finalize () 
```




<hr>



### function finish\_remaining\_tasks 

```C++
inline std::vector< ResultT > dynampi::HierarchicalLockFreeMPIWorkDistributor::finish_remaining_tasks () 
```




<hr>



### function insert\_task 

```C++
inline void dynampi::HierarchicalLockFreeMPIWorkDistributor::insert_task (
    TaskT task
) 
```




<hr>



### function insert\_tasks 

```C++
inline void dynampi::HierarchicalLockFreeMPIWorkDistributor::insert_tasks (
    const std::vector< TaskT > & tasks
) 
```




<hr>



### function is\_root\_manager 

```C++
inline bool dynampi::HierarchicalLockFreeMPIWorkDistributor::is_root_manager () const
```




<hr>



### function remaining\_tasks\_count 

```C++
inline size_t dynampi::HierarchicalLockFreeMPIWorkDistributor::remaining_tasks_count () const
```




<hr>



### function run\_tasks 

```C++
inline std::vector< ResultT > dynampi::HierarchicalLockFreeMPIWorkDistributor::run_tasks (
    const RunConfig & config=RunConfig {}
) 
```




<hr>



### function run\_worker 

```C++
inline void dynampi::HierarchicalLockFreeMPIWorkDistributor::run_worker () 
```




<hr>



### function ~HierarchicalLockFreeMPIWorkDistributor 

```C++
inline dynampi::HierarchicalLockFreeMPIWorkDistributor::~HierarchicalLockFreeMPIWorkDistributor () 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/hierarchical_lockfree_distributor.hpp`

