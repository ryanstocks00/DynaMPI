

# Namespace dynampi::detail



[**Namespace List**](namespaces.md) **>** [**dynampi**](namespacedynampi.md) **>** [**detail**](namespacedynampi_1_1detail.md)




















## Classes

| Type | Name |
| ---: | :--- |
| class | [**AsyncPutLevel**](classdynampi_1_1detail_1_1AsyncPutLevel.md) &lt;[**typename**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) [**TaskT**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes), [**typename**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) [**ResultT**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes)&gt;<br> |
| class | [**LockFreeLevel**](classdynampi_1_1detail_1_1LockFreeLevel.md) &lt;[**typename**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) [**TaskT**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes), [**typename**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) [**ResultT**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes)&gt;<br> |






















## Public Functions

| Type | Name |
| ---: | :--- |
|  int | [**mpi\_type\_size\_bytes**](#function-mpi_type_size_bytes) () <br> |
|  [**int64\_t**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**read\_i64**](#function-read_i64) ([**const**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) std::byte \* buffer, [**size\_t**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) buffer\_size, [**size\_t**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) offset) <br> |
|  [**void**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**read\_result\_bytes**](#function-read_result_bytes) ([**const**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) std::byte \* buffer, [**size\_t**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) buffer\_size, [**size\_t**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) offset, [**T**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) & value, [**size\_t**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) data\_bytes) <br> |
|  [**void**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**rma\_wait\_idle**](#function-rma_wait_idle) ([**MPI\_Win**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) window) <br> |
|  [**constexpr**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) [**size\_t**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**round\_up\_8**](#function-round_up_8) ([**size\_t**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) bytes) <br> |
|  [**void**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**write\_bytes**](#function-write_bytes) (std::byte \* buffer, [**size\_t**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) buffer\_size, [**size\_t**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) offset, [**const**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) [**void**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) \* src, [**size\_t**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) nbytes) <br> |
|  [**void**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) | [**write\_i64**](#function-write_i64) (std::byte \* buffer, [**size\_t**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) buffer\_size, [**size\_t**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) offset, [**int64\_t**](namespacedynampi_1_1detail.md#function-mpi_type_size_bytes) value) <br> |




























## Public Functions Documentation




### function mpi\_type\_size\_bytes 

```C++
template<typename T>
inline int dynampi::detail::mpi_type_size_bytes () 
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



### function rma\_wait\_idle 

```C++
inline void dynampi::detail::rma_wait_idle (
    MPI_Win window
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
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/hierarchical_async_put_lockfree_distributor.hpp`

