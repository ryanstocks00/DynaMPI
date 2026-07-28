

# Class dynampi::detail::LockFreeLevel

**template &lt;[**typename**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) [**TaskT**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes), [**typename**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) [**ResultT**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes)&gt;**



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**detail**](namespacedynampi_1_1detail.md) **>** [**LockFreeLevel**](classdynampi_1_1detail_1_1LockFreeLevel.md)





* `#include <hierarchical_lockfree_distributor.hpp>`















## Classes

| Type | Name |
| ---: | :--- |
| struct | [**Config**](structdynampi_1_1detail_1_1LockFreeLevel_1_1Config.md) <br> |






















## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**LockFreeLevel**](#function-lockfreelevel-12) ([**Config**](structdynampi_1_1detail_1_1LockFreeLevel_1_1Config.md) config) <br> |
|   | [**LockFreeLevel**](#function-lockfreelevel-22) ([**const**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) [**LockFreeLevel**](classdynampi_1_1detail_1_1LockFreeLevel.md) &) = delete<br> |
|  [**int**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**comm\_rank**](#function-comm_rank) () const<br> |
|  [**int**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**comm\_size**](#function-comm_size) () const<br> |
|  [**bool**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**drained**](#function-drained) () <br> |
|  [**bool**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**gather\_fully\_done**](#function-gather_fully_done) () <br> |
|  [**bool**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**has\_pending\_results**](#function-has_pending_results) () const<br> |
|  [**void**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**idle\_wait**](#function-idle_wait) () <br> |
|  [**bool**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**is\_owner**](#function-is_owner) () const<br> |
|  [**void**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**mark\_finished**](#function-mark_finished) () <br> |
|  [**void**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**mark\_gather\_done**](#function-mark_gather_done) () <br> |
|  [**bool**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**maybe\_participate\_in\_gather**](#function-maybe_participate_in_gather) () <br> |
|  [**LockFreeLevel**](classdynampi_1_1detail_1_1LockFreeLevel.md) & | [**operator=**](#function-operator) ([**const**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) [**LockFreeLevel**](classdynampi_1_1detail_1_1LockFreeLevel.md) &) = delete<br> |
|  [**size\_t**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**owner\_collected\_count**](#function-owner_collected_count) () const<br> |
|  [**bool**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**owner\_marked\_finished**](#function-owner_marked_finished) () const<br> |
|  [**size\_t**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**owner\_published\_count**](#function-owner_published_count) () const<br> |
|  [**void**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**publish\_tasks**](#function-publish_tasks) ([**const**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) std::vector&lt; [**TaskT**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) &gt; & tasks) <br> |
|  std::vector&lt; [**ResultT**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) &gt; | [**request\_gather**](#function-request_gather) () <br> |
|  std::vector&lt; [**ResultT**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) &gt; | [**request\_gather\_throttled**](#function-request_gather_throttled) () <br> |
|  [**void**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**stage\_result**](#function-stage_result) ([**ResultT**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) result) <br> |
|  std::vector&lt; [**TaskT**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) &gt; | [**try\_claim\_batch**](#function-try_claim_batch) ([**int**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) want) <br> |
|   | [**~LockFreeLevel**](#function-lockfreelevel) () <br> |




























## Public Functions Documentation




### function LockFreeLevel [1/2]

```C++
inline explicit dynampi::detail::LockFreeLevel::LockFreeLevel (
    Config config
) 
```




<hr>



### function LockFreeLevel [2/2]

```C++
dynampi::detail::LockFreeLevel::LockFreeLevel (
    const  LockFreeLevel &
) = delete
```




<hr>



### function comm\_rank 

```C++
inline int dynampi::detail::LockFreeLevel::comm_rank () const
```




<hr>



### function comm\_size 

```C++
inline int dynampi::detail::LockFreeLevel::comm_size () const
```




<hr>



### function drained 

```C++
inline bool dynampi::detail::LockFreeLevel::drained () 
```




<hr>



### function gather\_fully\_done 

```C++
inline bool dynampi::detail::LockFreeLevel::gather_fully_done () 
```




<hr>



### function has\_pending\_results 

```C++
inline bool dynampi::detail::LockFreeLevel::has_pending_results () const
```




<hr>



### function idle\_wait 

```C++
inline void dynampi::detail::LockFreeLevel::idle_wait () 
```




<hr>



### function is\_owner 

```C++
inline bool dynampi::detail::LockFreeLevel::is_owner () const
```




<hr>



### function mark\_finished 

```C++
inline void dynampi::detail::LockFreeLevel::mark_finished () 
```




<hr>



### function mark\_gather\_done 

```C++
inline void dynampi::detail::LockFreeLevel::mark_gather_done () 
```




<hr>



### function maybe\_participate\_in\_gather 

```C++
inline bool dynampi::detail::LockFreeLevel::maybe_participate_in_gather () 
```




<hr>



### function operator= 

```C++
LockFreeLevel & dynampi::detail::LockFreeLevel::operator= (
    const  LockFreeLevel &
) = delete
```




<hr>



### function owner\_collected\_count 

```C++
inline size_t dynampi::detail::LockFreeLevel::owner_collected_count () const
```




<hr>



### function owner\_marked\_finished 

```C++
inline bool dynampi::detail::LockFreeLevel::owner_marked_finished () const
```




<hr>



### function owner\_published\_count 

```C++
inline size_t dynampi::detail::LockFreeLevel::owner_published_count () const
```




<hr>



### function publish\_tasks 

```C++
inline void dynampi::detail::LockFreeLevel::publish_tasks (
    const std::vector< TaskT > & tasks
) 
```




<hr>



### function request\_gather 

```C++
inline std::vector< ResultT > dynampi::detail::LockFreeLevel::request_gather () 
```




<hr>



### function request\_gather\_throttled 

```C++
inline std::vector< ResultT > dynampi::detail::LockFreeLevel::request_gather_throttled () 
```




<hr>



### function stage\_result 

```C++
inline void dynampi::detail::LockFreeLevel::stage_result (
    ResultT result
) 
```




<hr>



### function try\_claim\_batch 

```C++
inline std::vector< TaskT > dynampi::detail::LockFreeLevel::try_claim_batch (
    int want
) 
```




<hr>



### function ~LockFreeLevel 

```C++
inline dynampi::detail::LockFreeLevel::~LockFreeLevel () 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/hierarchical_lockfree_distributor.hpp`

