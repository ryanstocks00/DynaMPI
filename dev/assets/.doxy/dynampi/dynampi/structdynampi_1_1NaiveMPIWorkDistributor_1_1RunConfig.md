

# Struct dynampi::NaiveMPIWorkDistributor::RunConfig



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**NaiveMPIWorkDistributor**](classdynampi_1_1NaiveMPIWorkDistributor.md) **>** [**RunConfig**](structdynampi_1_1NaiveMPIWorkDistributor_1_1RunConfig.md)





* `#include <naive_distributor.hpp>`





















## Public Attributes

| Type | Name |
| ---: | :--- |
|  [**bool**](structdynampi_1_1MPI__Type.md) | [**allow\_more\_than\_target\_tasks**](#variable-allow_more_than_target_tasks)   = `[**true**](structdynampi_1_1MPI__Type.md)`<br> |
|  std::optional&lt; [**double**](structdynampi_1_1MPI__Type.md) &gt; | [**max\_seconds**](#variable-max_seconds)   = `std::nullopt`<br> |
|  [**size\_t**](structdynampi_1_1MPI__Type.md) | [**target\_num\_tasks**](#variable-target_num_tasks)   = `std::numeric\_limits&lt;[**size\_t**](structdynampi_1_1MPI__Type.md)&gt;::max()`<br> |












































## Public Attributes Documentation




### variable allow\_more\_than\_target\_tasks 

```C++
bool dynampi::NaiveMPIWorkDistributor< TaskT, ResultT, Options >::RunConfig::allow_more_than_target_tasks;
```




<hr>



### variable max\_seconds 

```C++
std::optional<double> dynampi::NaiveMPIWorkDistributor< TaskT, ResultT, Options >::RunConfig::max_seconds;
```




<hr>



### variable target\_num\_tasks 

```C++
size_t dynampi::NaiveMPIWorkDistributor< TaskT, ResultT, Options >::RunConfig::target_num_tasks;
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/naive_distributor.hpp`

