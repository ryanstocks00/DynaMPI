

# Namespace dynampi::detail



[**Namespace List**](namespaces.md) **>** [**dynampi**](namespacedynampi.md) **>** [**detail**](namespacedynampi_1_1detail.md)




















## Classes

| Type | Name |
| ---: | :--- |
| class | [**LockFreeRMALevel**](classdynampi_1_1detail_1_1LockFreeRMALevel.md) &lt;[**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type)&gt;<br> |






















## Public Functions

| Type | Name |
| ---: | :--- |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**backoff\_sleep**](#function-backoff_sleep) (std::chrono::microseconds duration) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**check\_task\_capacity**](#function-check_task_capacity) ([**int64\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) start, [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) count, [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) max\_tasks, [**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**char**](namespacedynampi.md#function-check_fixed_size_mpi_type) \* distributor\_name) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**check\_variable\_batch\_read**](#function-check_variable_batch_read) ([**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) offset, [**uint64\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) nbytes, [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) buf\_size, [**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**char**](namespacedynampi.md#function-check_fixed_size_mpi_type) \* what) <br> |
|  [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**mpi\_type\_size\_bytes**](#function-mpi_type_size_bytes) () <br> |
|  std::vector&lt; std::byte &gt; | [**pack\_variable\_batch**](#function-pack_variable_batch) ([**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) std::vector&lt; [**T**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; & items) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**read\_bytes**](#function-read_bytes) ([**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) \* dst, [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) dst\_capacity, [**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) std::byte \* buffer, [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) buffer\_size, [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) offset, [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) nbytes) <br> |
|  [**int64\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**read\_i64**](#function-read_i64) ([**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) std::byte \* buffer, [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) buffer\_size, [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) offset) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**read\_result\_bytes**](#function-read_result_bytes) ([**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) std::byte \* buffer, [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) buffer\_size, [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) offset, [**T**](namespacedynampi.md#function-check_fixed_size_mpi_type) & value, [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) data\_bytes) <br> |
|  [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**resolve\_upper\_fanout**](#function-resolve_upper_fanout) ([**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) manager\_count, [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) configured\_fanout) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**rma\_wait\_idle**](#function-rma_wait_idle) ([**MPI\_Win**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**MPI\_Comm**](namespacedynampi.md#function-check_fixed_size_mpi_type) comm) <br> |
|  [**constexpr**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**round\_up\_8**](#function-round_up_8) ([**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) bytes) <br> |
|  std::optional&lt; [**Communicator**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; | [**split\_local\_worker\_communicator**](#function-split_local_worker_communicator) ([**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**Communicator**](namespacedynampi.md#function-check_fixed_size_mpi_type) & world\_comm, [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) manager\_rank, [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) max\_local\_group\_size) <br> |
|  [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**sum\_subtree\_widths\_to\_group\_leader**](#function-sum_subtree_widths_to_group_leader) ([**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) local\_width, [**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**Communicator**](namespacedynampi.md#function-check_fixed_size_mpi_type) & group\_comm) <br> |
|  std::vector&lt; [**T**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; | [**unpack\_variable\_batch**](#function-unpack_variable_batch) ([**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) std::vector&lt; std::byte &gt; & buf) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**write\_bytes**](#function-write_bytes) (std::byte \* buffer, [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) buffer\_size, [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) offset, [**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) \* src, [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) nbytes) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**write\_i64**](#function-write_i64) (std::byte \* buffer, [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) buffer\_size, [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) offset, [**int64\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) value) <br> |




























## Public Functions Documentation




### function backoff\_sleep 

```C++
inline void dynampi::detail::backoff_sleep (
    std::chrono::microseconds duration
) 
```




<hr>



### function check\_task\_capacity 

```C++
inline void dynampi::detail::check_task_capacity (
    int64_t start,
    size_t count,
    int max_tasks,
    const  char * distributor_name
) 
```




<hr>



### function check\_variable\_batch\_read 

```C++
inline void dynampi::detail::check_variable_batch_read (
    size_t offset,
    uint64_t nbytes,
    size_t buf_size,
    const  char * what
) 
```




<hr>



### function mpi\_type\_size\_bytes 

```C++
template<typename T>
inline int dynampi::detail::mpi_type_size_bytes () 
```




<hr>



### function pack\_variable\_batch 

```C++
template<typename T>
std::vector< std::byte > dynampi::detail::pack_variable_batch (
    const std::vector< T > & items
) 
```




<hr>



### function read\_bytes 

```C++
inline void dynampi::detail::read_bytes (
    void * dst,
    size_t dst_capacity,
    const std::byte * buffer,
    size_t buffer_size,
    size_t offset,
    size_t nbytes
) 
```




<hr>



### function read\_i64 

```C++
inline int64_t dynampi::detail::read_i64 (
    const std::byte * buffer,
    size_t buffer_size,
    size_t offset
) 
```




<hr>



### function read\_result\_bytes 

```C++
template<typename T>
inline void dynampi::detail::read_result_bytes (
    const std::byte * buffer,
    size_t buffer_size,
    size_t offset,
    T & value,
    size_t data_bytes
) 
```




<hr>



### function resolve\_upper\_fanout 

```C++
inline int dynampi::detail::resolve_upper_fanout (
    int manager_count,
    int configured_fanout
) 
```




<hr>



### function rma\_wait\_idle 

```C++
inline void dynampi::detail::rma_wait_idle (
    MPI_Win,
    MPI_Comm comm
) 
```




<hr>



### function round\_up\_8 

```C++
inline constexpr  size_t dynampi::detail::round_up_8 (
    size_t bytes
) 
```




<hr>



### function split\_local\_worker\_communicator 

```C++
template<typename Communicator>
std::optional< Communicator > dynampi::detail::split_local_worker_communicator (
    const  Communicator & world_comm,
    int manager_rank,
    int max_local_group_size
) 
```




<hr>



### function sum\_subtree\_widths\_to\_group\_leader 

```C++
template<typename Communicator>
int dynampi::detail::sum_subtree_widths_to_group_leader (
    int local_width,
    const  Communicator & group_comm
) 
```




<hr>



### function unpack\_variable\_batch 

```C++
template<typename T>
std::vector< T > dynampi::detail::unpack_variable_batch (
    const std::vector< std::byte > & buf
) 
```




<hr>



### function write\_bytes 

```C++
inline void dynampi::detail::write_bytes (
    std::byte * buffer,
    size_t buffer_size,
    size_t offset,
    const  void * src,
    size_t nbytes
) 
```




<hr>



### function write\_i64 

```C++
inline void dynampi::detail::write_i64 (
    std::byte * buffer,
    size_t buffer_size,
    size_t offset,
    int64_t value
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/hierarchical_lockfree_rma_distributor.hpp`

