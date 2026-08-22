

# Class dynampi::HierarchicalLockFreeRMAWorkDistributor

**template &lt;[**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type), typename... Options&gt;**



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**HierarchicalLockFreeRMAWorkDistributor**](classdynampi_1_1HierarchicalLockFreeRMAWorkDistributor.md)





* `#include <hierarchical_lockfree_rma_distributor.hpp>`















## Classes

| Type | Name |
| ---: | :--- |
| struct | [**Config**](structdynampi_1_1HierarchicalLockFreeRMAWorkDistributor_1_1Config.md) <br> |
| struct | [**RunConfig**](structdynampi_1_1HierarchicalLockFreeRMAWorkDistributor_1_1RunConfig.md) <br> |








## Public Static Attributes

| Type | Name |
| ---: | :--- |
|  [**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**ordered**](#variable-ordered)   = `[**true**](namespacedynampi.md#function-check_fixed_size_mpi_type)`<br> |














## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**HierarchicalLockFreeRMAWorkDistributor**](#function-hierarchicallockfreermaworkdistributor) (std::function&lt; [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type)([**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type))&gt; worker\_function, [**Config**](structdynampi_1_1HierarchicalLockFreeRMAWorkDistributor_1_1Config.md) config={}) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**finalize**](#function-finalize) () <br> |
|  std::vector&lt; [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; | [**finish\_remaining\_tasks**](#function-finish_remaining_tasks) () <br> |
|  std::vector&lt; [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; | [**gather\_once**](#function-gather_once) () <br> |
|  [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**has\_task\_errors**](#function-has_task_errors) () const<br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**insert\_task**](#function-insert_task) ([**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type) task) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**insert\_tasks**](#function-insert_tasks) ([**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) std::vector&lt; [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; & tasks) <br> |
|  [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**is\_root\_manager**](#function-is_root_manager) () const<br> |
|  [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**remaining\_tasks\_count**](#function-remaining_tasks_count) () const<br> |
|  std::vector&lt; [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; | [**run\_tasks**](#function-run_tasks) ([**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**RunConfig**](structdynampi_1_1HierarchicalLockFreeRMAWorkDistributor_1_1RunConfig.md) & config=[**RunConfig**](structdynampi_1_1HierarchicalLockFreeRMAWorkDistributor_1_1RunConfig.md){}) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**run\_worker**](#function-run_worker) () <br> |
|  std::vector&lt; [**TaskError**](structdynampi_1_1TaskError.md) &gt; | [**take\_task\_errors**](#function-take_task_errors) () <br> |
|   | [**~HierarchicalLockFreeRMAWorkDistributor**](#function-hierarchicallockfreermaworkdistributor) () <br> |




























## Public Static Attributes Documentation




### variable ordered 

```C++
const bool dynampi::HierarchicalLockFreeRMAWorkDistributor< TaskT, ResultT, Options >::ordered;
```




<hr>
## Public Functions Documentation




### function HierarchicalLockFreeRMAWorkDistributor 

```C++
inline explicit dynampi::HierarchicalLockFreeRMAWorkDistributor::HierarchicalLockFreeRMAWorkDistributor (
    std::function< ResultT ( TaskT )> worker_function,
    Config config={}
) 
```




<hr>



### function finalize 

```C++
inline void dynampi::HierarchicalLockFreeRMAWorkDistributor::finalize () 
```




<hr>



### function finish\_remaining\_tasks 

```C++
inline std::vector< ResultT > dynampi::HierarchicalLockFreeRMAWorkDistributor::finish_remaining_tasks () 
```




<hr>



### function gather\_once 

```C++
inline std::vector< ResultT > dynampi::HierarchicalLockFreeRMAWorkDistributor::gather_once () 
```




<hr>



### function has\_task\_errors 

```C++
inline bool dynampi::HierarchicalLockFreeRMAWorkDistributor::has_task_errors () const
```




<hr>



### function insert\_task 

```C++
inline void dynampi::HierarchicalLockFreeRMAWorkDistributor::insert_task (
    TaskT task
) 
```




<hr>



### function insert\_tasks 

```C++
inline void dynampi::HierarchicalLockFreeRMAWorkDistributor::insert_tasks (
    const std::vector< TaskT > & tasks
) 
```




<hr>



### function is\_root\_manager 

```C++
inline bool dynampi::HierarchicalLockFreeRMAWorkDistributor::is_root_manager () const
```




<hr>



### function remaining\_tasks\_count 

```C++
inline size_t dynampi::HierarchicalLockFreeRMAWorkDistributor::remaining_tasks_count () const
```




<hr>



### function run\_tasks 

```C++
inline std::vector< ResultT > dynampi::HierarchicalLockFreeRMAWorkDistributor::run_tasks (
    const  RunConfig & config=RunConfig {}
) 
```




<hr>



### function run\_worker 

```C++
inline void dynampi::HierarchicalLockFreeRMAWorkDistributor::run_worker () 
```




<hr>



### function take\_task\_errors 

```C++
inline std::vector< TaskError > dynampi::HierarchicalLockFreeRMAWorkDistributor::take_task_errors () 
```




<hr>



### function ~HierarchicalLockFreeRMAWorkDistributor 

```C++
inline dynampi::HierarchicalLockFreeRMAWorkDistributor::~HierarchicalLockFreeRMAWorkDistributor () 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/hierarchical_lockfree_rma_distributor.hpp`

