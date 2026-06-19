

# Namespace dynampi::detail



[**Namespace List**](namespaces.md) **>** [**dynampi**](namespacedynampi.md) **>** [**detail**](namespacedynampi_1_1detail.md)










































## Public Functions

| Type | Name |
| ---: | :--- |
|  [**int**](structdynampi_1_1MPI__Type.md) | [**mpi\_type\_size\_bytes**](#function-mpi_type_size_bytes) () <br> |
|  [**int64\_t**](structdynampi_1_1MPI__Type.md) | [**read\_i64**](#function-read_i64) ([**const**](structdynampi_1_1MPI__Type.md) std::byte \* buffer, [**size\_t**](structdynampi_1_1MPI__Type.md) buffer\_size, [**size\_t**](structdynampi_1_1MPI__Type.md) offset) <br> |
|  [**void**](structdynampi_1_1MPI__Type.md) | [**read\_result\_bytes**](#function-read_result_bytes) ([**const**](structdynampi_1_1MPI__Type.md) std::byte \* buffer, [**size\_t**](structdynampi_1_1MPI__Type.md) buffer\_size, [**size\_t**](structdynampi_1_1MPI__Type.md) offset, [**T**](structdynampi_1_1MPI__Type.md) & value, [**size\_t**](structdynampi_1_1MPI__Type.md) data\_bytes) <br> |
|  [**void**](structdynampi_1_1MPI__Type.md) | [**rma\_wait\_idle**](#function-rma_wait_idle) ([**MPI\_Win**](structdynampi_1_1MPI__Type.md) window) <br> |
|  [**constexpr**](structdynampi_1_1MPI__Type.md) [**size\_t**](structdynampi_1_1MPI__Type.md) | [**round\_up\_8**](#function-round_up_8) ([**size\_t**](structdynampi_1_1MPI__Type.md) bytes) <br> |
|  [**void**](structdynampi_1_1MPI__Type.md) | [**write\_bytes**](#function-write_bytes) (std::byte \* buffer, [**size\_t**](structdynampi_1_1MPI__Type.md) buffer\_size, [**size\_t**](structdynampi_1_1MPI__Type.md) offset, [**const**](structdynampi_1_1MPI__Type.md) [**void**](structdynampi_1_1MPI__Type.md) \* src, [**size\_t**](structdynampi_1_1MPI__Type.md) nbytes) <br> |
|  [**void**](structdynampi_1_1MPI__Type.md) | [**write\_i64**](#function-write_i64) (std::byte \* buffer, [**size\_t**](structdynampi_1_1MPI__Type.md) buffer\_size, [**size\_t**](structdynampi_1_1MPI__Type.md) offset, [**int64\_t**](structdynampi_1_1MPI__Type.md) value) <br> |




























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
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/lockfree_distributor.hpp`

