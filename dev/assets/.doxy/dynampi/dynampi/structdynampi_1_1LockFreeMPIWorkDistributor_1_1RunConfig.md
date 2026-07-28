

# Struct dynampi::LockFreeMPIWorkDistributor::RunConfig



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md) **>** [**RunConfig**](structdynampi_1_1LockFreeMPIWorkDistributor_1_1RunConfig.md)





* `#include <lockfree_distributor.hpp>`





















## Public Attributes

| Type | Name |
| ---: | :--- |
|  bool | [**allow\_more\_than\_target\_tasks**](#variable-allow_more_than_target_tasks)   = `true`<br> |
|  std::optional&lt; double &gt; | [**max\_seconds**](#variable-max_seconds)   = `std::nullopt`<br> |
|  size\_t | [**target\_num\_tasks**](#variable-target_num_tasks)   = `std::numeric\_limits&lt;size\_t&gt;::max()`<br> |












































## Public Attributes Documentation




### variable allow\_more\_than\_target\_tasks 

```C++
bool dynampi::LockFreeMPIWorkDistributor< TaskT, ResultT, Options >::RunConfig::allow_more_than_target_tasks;
```




<hr>



### variable max\_seconds 

```C++
std::optional<double> dynampi::LockFreeMPIWorkDistributor< TaskT, ResultT, Options >::RunConfig::max_seconds;
```




<hr>



### variable target\_num\_tasks 

```C++
size_t dynampi::LockFreeMPIWorkDistributor< TaskT, ResultT, Options >::RunConfig::target_num_tasks;
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/lockfree_distributor.hpp`

