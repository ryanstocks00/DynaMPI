

# Class dynampi::detail::TaskErrorLog



[**ClassList**](annotated.md) **>** [**TaskErrorLog**](classdynampi_1_1detail_1_1TaskErrorLog.md)










































## Public Functions

| Type | Name |
| ---: | :--- |
|  [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**empty**](#function-empty) () const<br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**record**](#function-record) ([**TaskError**](structdynampi_1_1TaskError.md) error) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**rethrow\_first\_if**](#function-rethrow_first_if) ([**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) enabled) <br> |
|  [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**size**](#function-size) () const<br> |
|  std::vector&lt; [**TaskError**](structdynampi_1_1TaskError.md) &gt; | [**take**](#function-take) () <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**warn\_if\_unreported**](#function-warn_if_unreported) ([**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**char**](namespacedynampi.md#function-check_fixed_size_mpi_type) \* distributor) noexcept const<br> |




























## Public Functions Documentation




### function empty 

```C++
inline bool TaskErrorLog::empty () const
```




<hr>



### function record 

```C++
inline void TaskErrorLog::record (
    TaskError error
) 
```




<hr>



### function rethrow\_first\_if 

```C++
inline void TaskErrorLog::rethrow_first_if (
    bool enabled
) 
```




<hr>



### function size 

```C++
inline size_t TaskErrorLog::size () const
```




<hr>



### function take 

```C++
inline std::vector< TaskError > TaskErrorLog::take () 
```




<hr>



### function warn\_if\_unreported 

```C++
inline void TaskErrorLog::warn_if_unreported (
    const  char * distributor
) noexcept const
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/task_error.hpp`

