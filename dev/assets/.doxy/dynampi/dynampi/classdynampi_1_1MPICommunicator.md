

# Class dynampi::MPICommunicator

**template &lt;typename... Options&gt;**



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**MPICommunicator**](classdynampi_1_1MPICommunicator.md)





* `#include <mpi_communicator.hpp>`

















## Public Types

| Type | Name |
| ---: | :--- |
| enum  | [**Ownership**](#enum-ownership)  <br> |




















## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**MPICommunicator**](#function-mpicommunicator-13) (MPI\_Comm comm, [**Ownership**](classdynampi_1_1MPICommunicator.md#enum-ownership) ownership=Duplicate) <br> |
|   | [**MPICommunicator**](#function-mpicommunicator-23) (const [**MPICommunicator**](classdynampi_1_1MPICommunicator.md) & other) = delete<br> |
|   | [**MPICommunicator**](#function-mpicommunicator-33) ([**MPICommunicator**](classdynampi_1_1MPICommunicator.md) && other) noexcept<br> |
|  void | [**broadcast**](#function-broadcast) (T & data, int root=0) <br> |
|  void | [**gather**](#function-gather) (const T & data, std::vector&lt; T &gt; \* result, int root=0) <br> |
|  MPI\_Comm | [**get**](#function-get) () const<br> |
|  [**MPIGroup**](classdynampi_1_1MPIGroup.md) | [**get\_group**](#function-get_group) () const<br> |
|  const [**CommStatistics**](structdynampi_1_1CommStatistics.md) & | [**get\_statistics**](#function-get_statistics) () const<br> |
|  std::optional&lt; MPI\_Status &gt; | [**iprobe**](#function-iprobe) (int source=MPI\_ANY\_SOURCE, int tag=MPI\_ANY\_TAG) <br> |
|   | [**operator MPI\_Comm**](#function-operator-mpi_comm) () const<br> |
|  [**MPICommunicator**](classdynampi_1_1MPICommunicator.md) & | [**operator=**](#function-operator) (const [**MPICommunicator**](classdynampi_1_1MPICommunicator.md) & other) = delete<br> |
|  [**MPICommunicator**](classdynampi_1_1MPICommunicator.md) & | [**operator=**](#function-operator_1) ([**MPICommunicator**](classdynampi_1_1MPICommunicator.md) && other) = delete<br> |
|  MPI\_Status | [**probe**](#function-probe) (int source=MPI\_ANY\_SOURCE, int tag=MPI\_ANY\_TAG) <br> |
|  int | [**rank**](#function-rank) () const<br> |
|  void | [**recv**](#function-recv) (T & data, int source, int tag=0) <br> |
|  MPI\_Status | [**recv\_any**](#function-recv_any) (T & data, int source=MPI\_ANY\_SOURCE, int tag=MPI\_ANY\_TAG) <br> |
|  void | [**recv\_empty**](#function-recv_empty) (int source, int tag=0) <br>_Receives 0 elements of type T. Use when the sender used send\_empty&lt;T&gt;._  |
|  void | [**recv\_empty\_message**](#function-recv_empty_message) (int source, int tag=0) <br> |
|  void | [**send**](#function-send) (const T & data, int dest, int tag=0) <br> |
|  void | [**send\_empty**](#function-send_empty) (int dest, int tag=0) <br> |
|  int | [**size**](#function-size) () const<br> |
|  std::optional&lt; [**MPICommunicator**](classdynampi_1_1MPICommunicator.md) &gt; | [**split**](#function-split) (int color, int key=0) const<br> |
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
## Public Functions Documentation




### function MPICommunicator [1/3]

```C++
inline dynampi::MPICommunicator::MPICommunicator (
    MPI_Comm comm,
    Ownership ownership=Duplicate
) 
```




<hr>



### function MPICommunicator [2/3]

```C++
dynampi::MPICommunicator::MPICommunicator (
    const MPICommunicator & other
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
template<typename T>
inline void dynampi::MPICommunicator::broadcast (
    T & data,
    int root=0
) 
```




<hr>



### function gather 

```C++
template<typename T>
inline void dynampi::MPICommunicator::gather (
    const T & data,
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



### function get\_group 

```C++
inline MPIGroup dynampi::MPICommunicator::get_group () const
```




<hr>



### function get\_statistics 

```C++
inline const CommStatistics & dynampi::MPICommunicator::get_statistics () const
```




<hr>



### function iprobe 

```C++
inline std::optional< MPI_Status > dynampi::MPICommunicator::iprobe (
    int source=MPI_ANY_SOURCE,
    int tag=MPI_ANY_TAG
) 
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
    const MPICommunicator & other
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



### function rank 

```C++
inline int dynampi::MPICommunicator::rank () const
```




<hr>



### function recv 

```C++
template<typename T>
inline void dynampi::MPICommunicator::recv (
    T & data,
    int source,
    int tag=0
) 
```




<hr>



### function recv\_any 

```C++
template<typename T>
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
template<typename T>
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
template<typename T>
inline void dynampi::MPICommunicator::send (
    const T & data,
    int dest,
    int tag=0
) 
```




<hr>



### function send\_empty 

```C++
template<typename T>
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

