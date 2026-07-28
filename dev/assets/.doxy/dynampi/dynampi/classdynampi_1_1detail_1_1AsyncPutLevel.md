

# Class dynampi::detail::AsyncPutLevel

**template &lt;[**typename**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) [**TaskT**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes), [**typename**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) [**ResultT**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes)&gt;**



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**detail**](namespacedynampi_1_1detail.md) **>** [**AsyncPutLevel**](classdynampi_1_1detail_1_1AsyncPutLevel.md)





* `#include <hierarchical_async_put_lockfree_distributor.hpp>`















## Classes

| Type | Name |
| ---: | :--- |
| struct | [**ClaimedRange**](structdynampi_1_1detail_1_1AsyncPutLevel_1_1ClaimedRange.md) <br> |
| struct | [**Config**](structdynampi_1_1detail_1_1AsyncPutLevel_1_1Config.md) <br> |






















## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**AsyncPutLevel**](#function-asyncputlevel-12) ([**Config**](structdynampi_1_1detail_1_1AsyncPutLevel_1_1Config.md) config) <br> |
|   | [**AsyncPutLevel**](#function-asyncputlevel-22) ([**const**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) [**AsyncPutLevel**](classdynampi_1_1detail_1_1AsyncPutLevel.md) &) = delete<br> |
|  [**bool**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**check\_finished**](#function-check_finished) () <br> |
|  [**int**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**claim\_batch\_size**](#function-claim_batch_size) () const<br> |
|  [**int**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**comm\_rank**](#function-comm_rank) () const<br> |
|  [**int**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**comm\_size**](#function-comm_size) () const<br> |
|  [**bool**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**drained**](#function-drained) () <br> |
|  std::vector&lt; [**ResultT**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) &gt; | [**harvest\_ready\_results**](#function-harvest_ready_results) () <br> |
|  std::vector&lt; [**ResultT**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) &gt; | [**harvest\_ready\_results\_throttled**](#function-harvest_ready_results_throttled) () <br> |
|  [**void**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**idle\_wait**](#function-idle_wait) () <br> |
|  [**bool**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**is\_owner**](#function-is_owner) () const<br> |
|  [**void**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**mark\_finished**](#function-mark_finished) () <br> |
|  [**AsyncPutLevel**](classdynampi_1_1detail_1_1AsyncPutLevel.md) & | [**operator=**](#function-operator) ([**const**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) [**AsyncPutLevel**](classdynampi_1_1detail_1_1AsyncPutLevel.md) &) = delete<br> |
|  [**size\_t**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**owner\_collected\_count**](#function-owner_collected_count) () const<br> |
|  [**bool**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**owner\_marked\_finished**](#function-owner_marked_finished) () const<br> |
|  [**size\_t**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**owner\_published\_count**](#function-owner_published_count) () const<br> |
|  [**void**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**publish\_tasks**](#function-publish_tasks) ([**const**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) std::vector&lt; [**TaskT**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) &gt; & tasks) <br> |
|  [**ClaimedRange**](structdynampi_1_1detail_1_1AsyncPutLevel_1_1ClaimedRange.md) | [**try\_claim**](#function-try_claim) () <br> |
|  [**void**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**write\_result\_range**](#function-write_result_range) ([**int64\_t**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) start, [**const**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) std::vector&lt; [**ResultT**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) &gt; & results) <br> |
|   | [**~AsyncPutLevel**](#function-asyncputlevel) () <br> |




























## Public Functions Documentation




### function AsyncPutLevel [1/2]

```C++
inline explicit dynampi::detail::AsyncPutLevel::AsyncPutLevel (
    Config config
) 
```




<hr>



### function AsyncPutLevel [2/2]

```C++
dynampi::detail::AsyncPutLevel::AsyncPutLevel (
    const  AsyncPutLevel &
) = delete
```




<hr>



### function check\_finished 

```C++
inline bool dynampi::detail::AsyncPutLevel::check_finished () 
```




<hr>



### function claim\_batch\_size 

```C++
inline int dynampi::detail::AsyncPutLevel::claim_batch_size () const
```




<hr>



### function comm\_rank 

```C++
inline int dynampi::detail::AsyncPutLevel::comm_rank () const
```




<hr>



### function comm\_size 

```C++
inline int dynampi::detail::AsyncPutLevel::comm_size () const
```




<hr>



### function drained 

```C++
inline bool dynampi::detail::AsyncPutLevel::drained () 
```




<hr>



### function harvest\_ready\_results 

```C++
inline std::vector< ResultT > dynampi::detail::AsyncPutLevel::harvest_ready_results () 
```




<hr>



### function harvest\_ready\_results\_throttled 

```C++
inline std::vector< ResultT > dynampi::detail::AsyncPutLevel::harvest_ready_results_throttled () 
```




<hr>



### function idle\_wait 

```C++
inline void dynampi::detail::AsyncPutLevel::idle_wait () 
```




<hr>



### function is\_owner 

```C++
inline bool dynampi::detail::AsyncPutLevel::is_owner () const
```




<hr>



### function mark\_finished 

```C++
inline void dynampi::detail::AsyncPutLevel::mark_finished () 
```




<hr>



### function operator= 

```C++
AsyncPutLevel & dynampi::detail::AsyncPutLevel::operator= (
    const  AsyncPutLevel &
) = delete
```




<hr>



### function owner\_collected\_count 

```C++
inline size_t dynampi::detail::AsyncPutLevel::owner_collected_count () const
```




<hr>



### function owner\_marked\_finished 

```C++
inline bool dynampi::detail::AsyncPutLevel::owner_marked_finished () const
```




<hr>



### function owner\_published\_count 

```C++
inline size_t dynampi::detail::AsyncPutLevel::owner_published_count () const
```




<hr>



### function publish\_tasks 

```C++
inline void dynampi::detail::AsyncPutLevel::publish_tasks (
    const std::vector< TaskT > & tasks
) 
```




<hr>



### function try\_claim 

```C++
inline ClaimedRange dynampi::detail::AsyncPutLevel::try_claim () 
```




<hr>



### function write\_result\_range 

```C++
inline void dynampi::detail::AsyncPutLevel::write_result_range (
    int64_t start,
    const std::vector< ResultT > & results
) 
```




<hr>



### function ~AsyncPutLevel 

```C++
inline dynampi::detail::AsyncPutLevel::~AsyncPutLevel () 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/hierarchical_async_put_lockfree_distributor.hpp`

