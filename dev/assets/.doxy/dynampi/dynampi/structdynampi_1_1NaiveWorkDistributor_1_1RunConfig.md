

# Struct dynampi::NaiveWorkDistributor::RunConfig



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**NaiveWorkDistributor**](classdynampi_1_1NaiveWorkDistributor.md) **>** [**RunConfig**](structdynampi_1_1NaiveWorkDistributor_1_1RunConfig.md)





* `#include <naive_distributor.hpp>`





















## Public Attributes

| Type | Name |
| ---: | :--- |
|  [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**allow\_more\_than\_target\_tasks**](#variable-allow_more_than_target_tasks)   = `[**true**](namespacedynampi.md#function-check_fixed_size_mpi_type)`<br> |
|  std::optional&lt; [**double**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; | [**max\_seconds**](#variable-max_seconds)   = `std::nullopt`<br> |
|  [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**target\_num\_tasks**](#variable-target_num_tasks)   = `std::numeric\_limits&lt;[**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type)&gt;[**::max**](namespacedynampi.md#function-check_fixed_size_mpi_type)()`<br> |












































## Public Attributes Documentation




### variable allow\_more\_than\_target\_tasks 

```C++
bool dynampi::NaiveWorkDistributor< TaskT, ResultT, Options >::RunConfig::allow_more_than_target_tasks;
```




<hr>



### variable max\_seconds 

```C++
std::optional<double> dynampi::NaiveWorkDistributor< TaskT, ResultT, Options >::RunConfig::max_seconds;
```




<hr>



### variable target\_num\_tasks 

```C++
size_t dynampi::NaiveWorkDistributor< TaskT, ResultT, Options >::RunConfig::target_num_tasks;
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/naive_distributor.hpp`

