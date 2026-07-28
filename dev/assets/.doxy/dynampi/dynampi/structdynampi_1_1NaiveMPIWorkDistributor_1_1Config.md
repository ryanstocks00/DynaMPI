

# Struct dynampi::NaiveMPIWorkDistributor::Config



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**NaiveMPIWorkDistributor**](classdynampi_1_1NaiveMPIWorkDistributor.md) **>** [**Config**](structdynampi_1_1NaiveMPIWorkDistributor_1_1Config.md)





* `#include <naive_distributor.hpp>`





















## Public Attributes

| Type | Name |
| ---: | :--- |
|  bool | [**auto\_run\_workers**](#variable-auto_run_workers)   = `true`<br> |
|  MPI\_Comm | [**comm**](#variable-comm)   = `MPI\_COMM\_WORLD`<br> |
|  int | [**manager\_rank**](#variable-manager_rank)   = `0`<br> |
|  int | [**max\_result\_size**](#variable-max_result_size)   = `1024`<br> |
|  bool | [**use\_immediate\_recv**](#variable-use_immediate_recv)   = `false`<br> |












































## Public Attributes Documentation




### variable auto\_run\_workers 

```C++
bool dynampi::NaiveMPIWorkDistributor< TaskT, ResultT, Options >::Config::auto_run_workers;
```




<hr>



### variable comm 

```C++
MPI_Comm dynampi::NaiveMPIWorkDistributor< TaskT, ResultT, Options >::Config::comm;
```




<hr>



### variable manager\_rank 

```C++
int dynampi::NaiveMPIWorkDistributor< TaskT, ResultT, Options >::Config::manager_rank;
```




<hr>



### variable max\_result\_size 

```C++
int dynampi::NaiveMPIWorkDistributor< TaskT, ResultT, Options >::Config::max_result_size;
```




<hr>



### variable use\_immediate\_recv 

```C++
bool dynampi::NaiveMPIWorkDistributor< TaskT, ResultT, Options >::Config::use_immediate_recv;
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/naive_distributor.hpp`

