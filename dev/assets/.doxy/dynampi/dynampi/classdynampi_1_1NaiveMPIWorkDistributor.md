

# Class dynampi::NaiveMPIWorkDistributor

**template &lt;[**typename**](structdynampi_1_1MPI__Type.md) [**TaskT**](structdynampi_1_1MPI__Type.md), [**typename**](structdynampi_1_1MPI__Type.md) [**ResultT**](structdynampi_1_1MPI__Type.md), typename... Options&gt;**



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**NaiveMPIWorkDistributor**](classdynampi_1_1NaiveMPIWorkDistributor.md)





* `#include <naive_distributor.hpp>`















## Classes

| Type | Name |
| ---: | :--- |
| struct | [**Config**](structdynampi_1_1NaiveMPIWorkDistributor_1_1Config.md) <br> |
| struct | [**RunConfig**](structdynampi_1_1NaiveMPIWorkDistributor_1_1RunConfig.md) <br> |
| struct | [**Statistics**](structdynampi_1_1NaiveMPIWorkDistributor_1_1Statistics.md) <br> |


## Public Types

| Type | Name |
| ---: | :--- |
| typedef std::conditional\_t&lt; statistics\_mode==StatisticsMode::Detailed, [**Statistics**](structdynampi_1_1NaiveMPIWorkDistributor_1_1Statistics.md), std::monostate &gt; | [**StatisticsT**](#typedef-statisticst)  <br> |






## Public Static Attributes

| Type | Name |
| ---: | :--- |
|  [**const**](structdynampi_1_1MPI__Type.md) [**bool**](structdynampi_1_1MPI__Type.md) | [**ordered**](#variable-ordered)   = `[**true**](structdynampi_1_1MPI__Type.md)`<br> |














## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**NaiveMPIWorkDistributor**](#function-naivempiworkdistributor) (std::function&lt; [**ResultT**](structdynampi_1_1MPI__Type.md)([**TaskT**](structdynampi_1_1MPI__Type.md))&gt; worker\_function, [**Config**](structdynampi_1_1NaiveMPIWorkDistributor_1_1Config.md) runtime\_config=[**Config**](structdynampi_1_1NaiveMPIWorkDistributor_1_1Config.md){}) <br> |
|  [**void**](structdynampi_1_1MPI__Type.md) | [**finalize**](#function-finalize) () <br> |
|  std::vector&lt; [**ResultT**](structdynampi_1_1MPI__Type.md) &gt; | [**finish\_remaining\_tasks**](#function-finish_remaining_tasks) () <br> |
|  [**const**](structdynampi_1_1MPI__Type.md) [**StatisticsT**](classdynampi_1_1NaiveMPIWorkDistributor.md#typedef-statisticst) & | [**get\_statistics**](#function-get_statistics) () const<br> |
|  [**void**](structdynampi_1_1MPI__Type.md) | [**insert\_task**](#function-insert_task-12) ([**TaskT**](structdynampi_1_1MPI__Type.md) task) <br> |
|  [**void**](structdynampi_1_1MPI__Type.md) | [**insert\_task**](#function-insert_task-22) ([**const**](structdynampi_1_1MPI__Type.md) [**TaskT**](structdynampi_1_1MPI__Type.md) & task, [**double**](structdynampi_1_1MPI__Type.md) priority) <br> |
|  [**void**](structdynampi_1_1MPI__Type.md) | [**insert\_tasks**](#function-insert_tasks) ([**const**](structdynampi_1_1MPI__Type.md) std::vector&lt; [**TaskT**](structdynampi_1_1MPI__Type.md) &gt; & tasks) <br> |
|  [**bool**](structdynampi_1_1MPI__Type.md) | [**is\_root\_manager**](#function-is_root_manager) () const<br> |
|  [**size\_t**](structdynampi_1_1MPI__Type.md) | [**remaining\_tasks\_count**](#function-remaining_tasks_count) () const<br> |
|  std::vector&lt; [**ResultT**](structdynampi_1_1MPI__Type.md) &gt; | [**run\_tasks**](#function-run_tasks) ([**RunConfig**](structdynampi_1_1NaiveMPIWorkDistributor_1_1RunConfig.md) config=[**RunConfig**](structdynampi_1_1NaiveMPIWorkDistributor_1_1RunConfig.md){}) <br> |
|  [**void**](structdynampi_1_1MPI__Type.md) | [**run\_worker**](#function-run_worker) () <br> |
|   | [**~NaiveMPIWorkDistributor**](#function-naivempiworkdistributor) () <br> |




























## Public Types Documentation




### typedef StatisticsT 

```C++
using dynampi::NaiveMPIWorkDistributor< TaskT, ResultT, Options >::StatisticsT =  std::conditional_t<statistics_mode == StatisticsMode::Detailed, Statistics, std::monostate>;
```




<hr>
## Public Static Attributes Documentation




### variable ordered 

```C++
const bool dynampi::NaiveMPIWorkDistributor< TaskT, ResultT, Options >::ordered;
```




<hr>
## Public Functions Documentation




### function NaiveMPIWorkDistributor 

```C++
inline explicit dynampi::NaiveMPIWorkDistributor::NaiveMPIWorkDistributor (
    std::function< ResultT ( TaskT )> worker_function,
    Config runtime_config=Config {}
) 
```




<hr>



### function finalize 

```C++
inline void dynampi::NaiveMPIWorkDistributor::finalize () 
```




<hr>



### function finish\_remaining\_tasks 

```C++
inline std::vector< ResultT > dynampi::NaiveMPIWorkDistributor::finish_remaining_tasks () 
```




<hr>



### function get\_statistics 

```C++
inline const  StatisticsT & dynampi::NaiveMPIWorkDistributor::get_statistics () const
```




<hr>



### function insert\_task [1/2]

```C++
inline void dynampi::NaiveMPIWorkDistributor::insert_task (
    TaskT task
) 
```




<hr>



### function insert\_task [2/2]

```C++
inline void dynampi::NaiveMPIWorkDistributor::insert_task (
    const  TaskT & task,
    double priority
) 
```




<hr>



### function insert\_tasks 

```C++
inline void dynampi::NaiveMPIWorkDistributor::insert_tasks (
    const std::vector< TaskT > & tasks
) 
```




<hr>



### function is\_root\_manager 

```C++
inline bool dynampi::NaiveMPIWorkDistributor::is_root_manager () const
```




<hr>



### function remaining\_tasks\_count 

```C++
inline size_t dynampi::NaiveMPIWorkDistributor::remaining_tasks_count () const
```




<hr>



### function run\_tasks 

```C++
inline std::vector< ResultT > dynampi::NaiveMPIWorkDistributor::run_tasks (
    RunConfig config=RunConfig {}
) 
```




<hr>



### function run\_worker 

```C++
inline void dynampi::NaiveMPIWorkDistributor::run_worker () 
```




<hr>



### function ~NaiveMPIWorkDistributor 

```C++
inline dynampi::NaiveMPIWorkDistributor::~NaiveMPIWorkDistributor () 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/naive_distributor.hpp`

