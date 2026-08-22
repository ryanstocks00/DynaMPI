

# Class dynampi::detail::LockFreeRMALevel

**template &lt;[**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type)&gt;**



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**detail**](namespacedynampi_1_1detail.md) **>** [**LockFreeRMALevel**](classdynampi_1_1detail_1_1LockFreeRMALevel.md)





* `#include <hierarchical_lockfree_rma_distributor.hpp>`















## Classes

| Type | Name |
| ---: | :--- |
| struct | [**ClaimedRange**](structdynampi_1_1detail_1_1LockFreeRMALevel_1_1ClaimedRange.md) <br> |
| struct | [**Config**](structdynampi_1_1detail_1_1LockFreeRMALevel_1_1Config.md) <br> |






















## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**LockFreeRMALevel**](#function-lockfreermalevel-12) ([**Config**](structdynampi_1_1detail_1_1LockFreeRMALevel_1_1Config.md) config, [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) claim\_width=1) <br> |
|   | [**LockFreeRMALevel**](#function-lockfreermalevel-22) ([**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**LockFreeRMALevel**](classdynampi_1_1detail_1_1LockFreeRMALevel.md) &) = delete<br> |
|  [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**check\_finished**](#function-check_finished) () <br> |
|  [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**claim\_width**](#function-claim_width) () const<br> |
|  [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**comm\_rank**](#function-comm_rank) () const<br> |
|  [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**comm\_size**](#function-comm_size) () const<br> |
|  [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**drained**](#function-drained) () <br> |
|  std::vector&lt; [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; | [**harvest\_ready\_results**](#function-harvest_ready_results) () <br> |
|  std::vector&lt; [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; | [**harvest\_ready\_results\_throttled**](#function-harvest_ready_results_throttled) () <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**idle\_wait**](#function-idle_wait) () <br> |
|  [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**is\_owner**](#function-is_owner) () const<br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**mark\_finished**](#function-mark_finished) () <br> |
|  [**LockFreeRMALevel**](classdynampi_1_1detail_1_1LockFreeRMALevel.md) & | [**operator=**](#function-operator) ([**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**LockFreeRMALevel**](classdynampi_1_1detail_1_1LockFreeRMALevel.md) &) = delete<br> |
|  [**int64\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**owner\_available\_capacity**](#function-owner_available_capacity) () const<br> |
|  [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**owner\_collected\_count**](#function-owner_collected_count) () const<br> |
|  [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**owner\_marked\_finished**](#function-owner_marked_finished) () const<br> |
|  [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**owner\_published\_count**](#function-owner_published_count) () const<br> |
|  [**int64\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**owner\_unordered\_completed\_estimate**](#function-owner_unordered_completed_estimate) () const<br> |
|  [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**publish\_tasks**](#function-publish_tasks) ([**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) std::vector&lt; [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; & tasks) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**report\_task\_error**](#function-report_task_error) ([**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**TaskError**](structdynampi_1_1TaskError.md) & error) <br> |
|  std::vector&lt; [**TaskError**](structdynampi_1_1TaskError.md) &gt; | [**take\_errors**](#function-take_errors) () <br> |
|  [**ClaimedRange**](structdynampi_1_1detail_1_1LockFreeRMALevel_1_1ClaimedRange.md) | [**try\_claim**](#function-try_claim) () <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**write\_result\_range**](#function-write_result_range) ([**int64\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) start, [**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) std::vector&lt; [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; & results, [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) contains\_error=[**false**](namespacedynampi.md#function-check_fixed_size_mpi_type)) <br> |
|   | [**~LockFreeRMALevel**](#function-lockfreermalevel) () <br> |




























## Public Functions Documentation




### function LockFreeRMALevel [1/2]

```C++
inline explicit dynampi::detail::LockFreeRMALevel::LockFreeRMALevel (
    Config config,
    int claim_width=1
) 
```




<hr>



### function LockFreeRMALevel [2/2]

```C++
dynampi::detail::LockFreeRMALevel::LockFreeRMALevel (
    const  LockFreeRMALevel &
) = delete
```




<hr>



### function check\_finished 

```C++
inline bool dynampi::detail::LockFreeRMALevel::check_finished () 
```




<hr>



### function claim\_width 

```C++
inline int dynampi::detail::LockFreeRMALevel::claim_width () const
```




<hr>



### function comm\_rank 

```C++
inline int dynampi::detail::LockFreeRMALevel::comm_rank () const
```




<hr>



### function comm\_size 

```C++
inline int dynampi::detail::LockFreeRMALevel::comm_size () const
```




<hr>



### function drained 

```C++
inline bool dynampi::detail::LockFreeRMALevel::drained () 
```




<hr>



### function harvest\_ready\_results 

```C++
inline std::vector< ResultT > dynampi::detail::LockFreeRMALevel::harvest_ready_results () 
```




<hr>



### function harvest\_ready\_results\_throttled 

```C++
inline std::vector< ResultT > dynampi::detail::LockFreeRMALevel::harvest_ready_results_throttled () 
```




<hr>



### function idle\_wait 

```C++
inline void dynampi::detail::LockFreeRMALevel::idle_wait () 
```




<hr>



### function is\_owner 

```C++
inline bool dynampi::detail::LockFreeRMALevel::is_owner () const
```




<hr>



### function mark\_finished 

```C++
inline void dynampi::detail::LockFreeRMALevel::mark_finished () 
```




<hr>



### function operator= 

```C++
LockFreeRMALevel & dynampi::detail::LockFreeRMALevel::operator= (
    const  LockFreeRMALevel &
) = delete
```




<hr>



### function owner\_available\_capacity 

```C++
inline int64_t dynampi::detail::LockFreeRMALevel::owner_available_capacity () const
```




<hr>



### function owner\_collected\_count 

```C++
inline size_t dynampi::detail::LockFreeRMALevel::owner_collected_count () const
```




<hr>



### function owner\_marked\_finished 

```C++
inline bool dynampi::detail::LockFreeRMALevel::owner_marked_finished () const
```




<hr>



### function owner\_published\_count 

```C++
inline size_t dynampi::detail::LockFreeRMALevel::owner_published_count () const
```




<hr>



### function owner\_unordered\_completed\_estimate 

```C++
inline int64_t dynampi::detail::LockFreeRMALevel::owner_unordered_completed_estimate () const
```




<hr>



### function publish\_tasks 

```C++
inline bool dynampi::detail::LockFreeRMALevel::publish_tasks (
    const std::vector< TaskT > & tasks
) 
```




<hr>



### function report\_task\_error 

```C++
inline void dynampi::detail::LockFreeRMALevel::report_task_error (
    const  TaskError & error
) 
```




<hr>



### function take\_errors 

```C++
inline std::vector< TaskError > dynampi::detail::LockFreeRMALevel::take_errors () 
```




<hr>



### function try\_claim 

```C++
inline ClaimedRange dynampi::detail::LockFreeRMALevel::try_claim () 
```




<hr>



### function write\_result\_range 

```C++
inline void dynampi::detail::LockFreeRMALevel::write_result_range (
    int64_t start,
    const std::vector< ResultT > & results,
    bool contains_error=false
) 
```




<hr>



### function ~LockFreeRMALevel 

```C++
inline dynampi::detail::LockFreeRMALevel::~LockFreeRMALevel () 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/hierarchical_lockfree_rma_distributor.hpp`

