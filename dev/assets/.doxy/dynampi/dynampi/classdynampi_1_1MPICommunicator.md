

# Class dynampi::MPICommunicator

**template &lt;typename... Options&gt;**



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**MPICommunicator**](classdynampi_1_1MPICommunicator.md)





* `#include <mpi_communicator.hpp>`

















## Public Types

| Type | Name |
| ---: | :--- |
| enum  | [**Ownership**](#enum-ownership)  <br> |






## Public Static Attributes

| Type | Name |
| ---: | :--- |
|  [**constexpr**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**kMaxRmaChunkBytes**](#variable-kmaxrmachunkbytes)   = `[**static\_cast**](namespacedynampi.md#function-check_fixed_size_mpi_type)&lt;[**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type)&gt;(std::numeric\_limits&lt;[**int**](namespacedynampi.md#function-check_fixed_size_mpi_type)&gt;[**::max**](namespacedynampi.md#function-check_fixed_size_mpi_type)())`<br> |














## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**MPICommunicator**](#function-mpicommunicator-13) ([**MPI\_Comm**](namespacedynampi.md#function-check_fixed_size_mpi_type) comm, [**Ownership**](classdynampi_1_1MPICommunicator.md#enum-ownership) ownership=Duplicate) <br> |
|   | [**MPICommunicator**](#function-mpicommunicator-23) ([**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**MPICommunicator**](classdynampi_1_1MPICommunicator.md) & other) = delete<br> |
|   | [**MPICommunicator**](#function-mpicommunicator-33) ([**MPICommunicator**](classdynampi_1_1MPICommunicator.md) && other) noexcept<br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**broadcast**](#function-broadcast) ([**T**](namespacedynampi.md#function-check_fixed_size_mpi_type) & data, [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) root=0) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**fetch\_and\_op**](#function-fetch_and_op) ([**T**](namespacedynampi.md#function-check_fixed_size_mpi_type) & origin, [**T**](namespacedynampi.md#function-check_fixed_size_mpi_type) & result, [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) target\_rank, [**MPI\_Aint**](namespacedynampi.md#function-check_fixed_size_mpi_type) target\_disp, [**MPI\_Op**](namespacedynampi.md#function-check_fixed_size_mpi_type) op, [**MPI\_Win**](namespacedynampi.md#function-check_fixed_size_mpi_type) win) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**gather**](#function-gather) ([**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**T**](namespacedynampi.md#function-check_fixed_size_mpi_type) & data, std::vector&lt; [**T**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; \* result, [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) root=0) <br> |
|  [**MPI\_Comm**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**get**](#function-get) () const<br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**get\_bytes**](#function-get_bytes) ([**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) \* dst, [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) n, [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) target\_rank, [**MPI\_Aint**](namespacedynampi.md#function-check_fixed_size_mpi_type) target\_disp, [**MPI\_Win**](namespacedynampi.md#function-check_fixed_size_mpi_type) win) <br> |
|  [**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**CommStatistics**](structdynampi_1_1CommStatistics.md) & | [**get\_statistics**](#function-get_statistics) () const<br> |
|   | [**operator MPI\_Comm**](#function-operator-mpi_comm) () const<br> |
|  [**MPICommunicator**](classdynampi_1_1MPICommunicator.md) & | [**operator=**](#function-operator) ([**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**MPICommunicator**](classdynampi_1_1MPICommunicator.md) & other) = delete<br> |
|  [**MPICommunicator**](classdynampi_1_1MPICommunicator.md) & | [**operator=**](#function-operator_1) ([**MPICommunicator**](classdynampi_1_1MPICommunicator.md) && other) = delete<br> |
|  [**MPI\_Status**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**probe**](#function-probe) ([**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) source=[**MPI\_ANY\_SOURCE**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) tag=[**MPI\_ANY\_TAG**](namespacedynampi.md#function-check_fixed_size_mpi_type)) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**put\_bytes**](#function-put_bytes) ([**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) \* src, [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) n, [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) target\_rank, [**MPI\_Aint**](namespacedynampi.md#function-check_fixed_size_mpi_type) target\_disp, [**MPI\_Win**](namespacedynampi.md#function-check_fixed_size_mpi_type) win) <br> |
|  [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**rank**](#function-rank) () const<br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**recv**](#function-recv) ([**T**](namespacedynampi.md#function-check_fixed_size_mpi_type) & data, [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) source, [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) tag=0) <br> |
|  [**MPI\_Status**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**recv\_any**](#function-recv_any) ([**T**](namespacedynampi.md#function-check_fixed_size_mpi_type) & data, [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) source=[**MPI\_ANY\_SOURCE**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) tag=[**MPI\_ANY\_TAG**](namespacedynampi.md#function-check_fixed_size_mpi_type)) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**recv\_empty**](#function-recv_empty) ([**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) source, [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) tag=0) <br>_Receives 0 elements of type T. Use when the sender used send\_empty&lt;T&gt;._  |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**recv\_empty\_message**](#function-recv_empty_message) ([**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) source, [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) tag=0) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**send**](#function-send) ([**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**T**](namespacedynampi.md#function-check_fixed_size_mpi_type) & data, [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) dest, [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) tag=0) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**send\_empty**](#function-send_empty) ([**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) dest, [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) tag=0) <br> |
|  [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**size**](#function-size) () const<br> |
|  std::optional&lt; [**MPICommunicator**](classdynampi_1_1MPICommunicator.md) &gt; | [**split**](#function-split) ([**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) color, [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) key=0) const<br> |
|  [**MPICommunicator**](classdynampi_1_1MPICommunicator.md) | [**split\_by\_node**](#function-split_by_node) () const<br> |
|   | [**~MPICommunicator**](#function-mpicommunicator) () <br> |




























## Public Types Documentation




### enum Ownership 

```C++
enum dynampi::MPICommunicator::Ownership {
    Reference,
    Move,
    Duplicate
};
```




<hr>
## Public Static Attributes Documentation




### variable kMaxRmaChunkBytes 

```C++
constexpr size_t dynampi::MPICommunicator< Options >::kMaxRmaChunkBytes;
```




<hr>
## Public Functions Documentation




### function MPICommunicator [1/3]

```C++
inline explicit dynampi::MPICommunicator::MPICommunicator (
    MPI_Comm comm,
    Ownership ownership=Duplicate
) 
```




<hr>



### function MPICommunicator [2/3]

```C++
dynampi::MPICommunicator::MPICommunicator (
    const  MPICommunicator & other
) = delete
```




<hr>



### function MPICommunicator [3/3]

```C++
inline dynampi::MPICommunicator::MPICommunicator (
    MPICommunicator && other
) noexcept
```




<hr>



### function broadcast 

```C++
template<typename  T>
inline void dynampi::MPICommunicator::broadcast (
    T & data,
    int root=0
) 
```




<hr>



### function fetch\_and\_op 

```C++
template<typename  T>
inline void dynampi::MPICommunicator::fetch_and_op (
    T & origin,
    T & result,
    int target_rank,
    MPI_Aint target_disp,
    MPI_Op op,
    MPI_Win win
) 
```




<hr>



### function gather 

```C++
template<typename  T>
inline void dynampi::MPICommunicator::gather (
    const  T & data,
    std::vector< T > * result,
    int root=0
) 
```




<hr>



### function get 

```C++
inline MPI_Comm dynampi::MPICommunicator::get () const
```




<hr>



### function get\_bytes 

```C++
inline void dynampi::MPICommunicator::get_bytes (
    void * dst,
    size_t n,
    int target_rank,
    MPI_Aint target_disp,
    MPI_Win win
) 
```




<hr>



### function get\_statistics 

```C++
inline const  CommStatistics & dynampi::MPICommunicator::get_statistics () const
```




<hr>



### function operator MPI\_Comm 

```C++
inline dynampi::MPICommunicator::operator MPI_Comm () const
```




<hr>



### function operator= 

```C++
MPICommunicator & dynampi::MPICommunicator::operator= (
    const  MPICommunicator & other
) = delete
```




<hr>



### function operator= 

```C++
MPICommunicator & dynampi::MPICommunicator::operator= (
    MPICommunicator && other
) = delete
```




<hr>



### function probe 

```C++
inline MPI_Status dynampi::MPICommunicator::probe (
    int source=MPI_ANY_SOURCE,
    int tag=MPI_ANY_TAG
) 
```




<hr>



### function put\_bytes 

```C++
inline void dynampi::MPICommunicator::put_bytes (
    const  void * src,
    size_t n,
    int target_rank,
    MPI_Aint target_disp,
    MPI_Win win
) 
```




<hr>



### function rank 

```C++
inline int dynampi::MPICommunicator::rank () const
```




<hr>



### function recv 

```C++
template<typename  T>
inline void dynampi::MPICommunicator::recv (
    T & data,
    int source,
    int tag=0
) 
```




<hr>



### function recv\_any 

```C++
template<typename  T>
inline MPI_Status dynampi::MPICommunicator::recv_any (
    T & data,
    int source=MPI_ANY_SOURCE,
    int tag=MPI_ANY_TAG
) 
```




<hr>



### function recv\_empty 

_Receives 0 elements of type T. Use when the sender used send\_empty&lt;T&gt;._ 
```C++
template<typename  T>
inline void dynampi::MPICommunicator::recv_empty (
    int source,
    int tag=0
) 
```




<hr>



### function recv\_empty\_message 

```C++
inline void dynampi::MPICommunicator::recv_empty_message (
    int source,
    int tag=0
) 
```




<hr>



### function send 

```C++
template<typename  T>
inline void dynampi::MPICommunicator::send (
    const  T & data,
    int dest,
    int tag=0
) 
```




<hr>



### function send\_empty 

```C++
template<typename  T>
inline void dynampi::MPICommunicator::send_empty (
    int dest,
    int tag=0
) 
```



Sends 0 elements of type T (same type as recv buffer) so that recv\_any(T&) can receive any worker message (REQUEST or RESULT) into a single buffer type. 


        

<hr>



### function size 

```C++
inline int dynampi::MPICommunicator::size () const
```




<hr>



### function split 

```C++
inline std::optional< MPICommunicator > dynampi::MPICommunicator::split (
    int color,
    int key=0
) const
```




<hr>



### function split\_by\_node 

```C++
inline MPICommunicator dynampi::MPICommunicator::split_by_node () const
```




<hr>



### function ~MPICommunicator 

```C++
inline dynampi::MPICommunicator::~MPICommunicator () 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/mpi/mpi_communicator.hpp`

