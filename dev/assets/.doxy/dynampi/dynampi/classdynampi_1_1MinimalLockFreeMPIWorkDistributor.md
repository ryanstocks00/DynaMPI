

# Class dynampi::MinimalLockFreeMPIWorkDistributor

**template &lt;[**typename**](structdynampi_1_1MPI__Type.md) [**ResultT**](structdynampi_1_1MPI__Type.md)&gt;**



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**MinimalLockFreeMPIWorkDistributor**](classdynampi_1_1MinimalLockFreeMPIWorkDistributor.md)





* `#include <lockfree_distributor.hpp>`















## Classes

| Type | Name |
| ---: | :--- |
| struct | [**Config**](structdynampi_1_1MinimalLockFreeMPIWorkDistributor_1_1Config.md) <br> |






















## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**MinimalLockFreeMPIWorkDistributor**](#function-minimallockfreempiworkdistributor) (std::function&lt; [**ResultT**](structdynampi_1_1MPI__Type.md)([**size\_t**](structdynampi_1_1MPI__Type.md))&gt; worker\_function, [**Config**](structdynampi_1_1MinimalLockFreeMPIWorkDistributor_1_1Config.md) config={}) <br> |
|  [**bool**](structdynampi_1_1MPI__Type.md) | [**is\_root\_manager**](#function-is_root_manager) () const<br> |
|  std::vector&lt; [**ResultT**](structdynampi_1_1MPI__Type.md) &gt; | [**run**](#function-run) ([**size\_t**](structdynampi_1_1MPI__Type.md) n\_tasks) <br> |
|   | [**~MinimalLockFreeMPIWorkDistributor**](#function-minimallockfreempiworkdistributor) () <br> |




























## Public Functions Documentation




### function MinimalLockFreeMPIWorkDistributor 

```C++
inline explicit dynampi::MinimalLockFreeMPIWorkDistributor::MinimalLockFreeMPIWorkDistributor (
    std::function< ResultT ( size_t )> worker_function,
    Config config={}
) 
```




<hr>



### function is\_root\_manager 

```C++
inline bool dynampi::MinimalLockFreeMPIWorkDistributor::is_root_manager () const
```




<hr>



### function run 

```C++
inline std::vector< ResultT > dynampi::MinimalLockFreeMPIWorkDistributor::run (
    size_t n_tasks
) 
```




<hr>



### function ~MinimalLockFreeMPIWorkDistributor 

```C++
inline dynampi::MinimalLockFreeMPIWorkDistributor::~MinimalLockFreeMPIWorkDistributor () 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/lockfree_distributor.hpp`

