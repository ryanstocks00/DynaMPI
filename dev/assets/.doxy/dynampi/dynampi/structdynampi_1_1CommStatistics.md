

# Struct dynampi::CommStatistics



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**CommStatistics**](structdynampi_1_1CommStatistics.md)





* `#include <mpi_communicator.hpp>`





















## Public Attributes

| Type | Name |
| ---: | :--- |
|  [**size\_t**](structdynampi_1_1MPI__Type.md) | [**bytes\_received**](#variable-bytes_received)   = `0`<br> |
|  [**size\_t**](structdynampi_1_1MPI__Type.md) | [**bytes\_sent**](#variable-bytes_sent)   = `0`<br> |
|  [**int**](structdynampi_1_1MPI__Type.md) | [**collective\_count**](#variable-collective_count)   = `0`<br> |
|  [**int**](structdynampi_1_1MPI__Type.md) | [**recv\_count**](#variable-recv_count)   = `0`<br> |
|  [**double**](structdynampi_1_1MPI__Type.md) | [**recv\_time**](#variable-recv_time)   = `0.0`<br> |
|  [**int**](structdynampi_1_1MPI__Type.md) | [**send\_count**](#variable-send_count)   = `0`<br> |
|  [**double**](structdynampi_1_1MPI__Type.md) | [**send\_time**](#variable-send_time)   = `0.0`<br> |
















## Public Functions

| Type | Name |
| ---: | :--- |
|  [**double**](structdynampi_1_1MPI__Type.md) | [**average\_receive\_size**](#function-average_receive_size) () const<br> |
|  [**double**](structdynampi_1_1MPI__Type.md) | [**average\_send\_size**](#function-average_send_size) () const<br> |
|  [**void**](structdynampi_1_1MPI__Type.md) | [**reset**](#function-reset) () <br> |




























## Public Attributes Documentation




### variable bytes\_received 

```C++
size_t dynampi::CommStatistics::bytes_received;
```




<hr>



### variable bytes\_sent 

```C++
size_t dynampi::CommStatistics::bytes_sent;
```




<hr>



### variable collective\_count 

```C++
int dynampi::CommStatistics::collective_count;
```




<hr>



### variable recv\_count 

```C++
int dynampi::CommStatistics::recv_count;
```




<hr>



### variable recv\_time 

```C++
double dynampi::CommStatistics::recv_time;
```




<hr>



### variable send\_count 

```C++
int dynampi::CommStatistics::send_count;
```




<hr>



### variable send\_time 

```C++
double dynampi::CommStatistics::send_time;
```




<hr>
## Public Functions Documentation




### function average\_receive\_size 

```C++
inline double dynampi::CommStatistics::average_receive_size () const
```




<hr>



### function average\_send\_size 

```C++
inline double dynampi::CommStatistics::average_send_size () const
```




<hr>



### function reset 

```C++
inline void dynampi::CommStatistics::reset () 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/mpi/mpi_communicator.hpp`

