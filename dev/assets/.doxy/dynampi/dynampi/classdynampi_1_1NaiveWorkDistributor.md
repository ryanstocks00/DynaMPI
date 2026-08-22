

# Class dynampi::NaiveWorkDistributor

**template &lt;[**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type), typename... Options&gt;**



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**NaiveWorkDistributor**](classdynampi_1_1NaiveWorkDistributor.md)





* `#include <naive_distributor.hpp>`















## Classes

| Type | Name |
| ---: | :--- |
| struct | [**Config**](structdynampi_1_1NaiveWorkDistributor_1_1Config.md) <br> |
| struct | [**RunConfig**](structdynampi_1_1NaiveWorkDistributor_1_1RunConfig.md) <br> |
| struct | [**Statistics**](structdynampi_1_1NaiveWorkDistributor_1_1Statistics.md) <br> |


## Public Types

| Type | Name |
| ---: | :--- |
| typedef std::conditional\_t&lt; statistics\_mode !=StatisticsMode::None, [**Statistics**](structdynampi_1_1NaiveWorkDistributor_1_1Statistics.md), std::monostate &gt; | [**StatisticsT**](#typedef-statisticst)  <br> |






## Public Static Attributes

| Type | Name |
| ---: | :--- |
|  [**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**ordered**](#variable-ordered)   = `[**true**](namespacedynampi.md#function-check_fixed_size_mpi_type)`<br> |














## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**NaiveWorkDistributor**](#function-naiveworkdistributor) (std::function&lt; [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type)([**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type))&gt; worker\_function, [**Config**](structdynampi_1_1NaiveWorkDistributor_1_1Config.md) runtime\_config=[**Config**](structdynampi_1_1NaiveWorkDistributor_1_1Config.md){}) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**finalize**](#function-finalize) () <br> |
|  std::vector&lt; [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; | [**finish\_remaining\_tasks**](#function-finish_remaining_tasks) () <br> |
|  [**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**StatisticsT**](classdynampi_1_1NaiveWorkDistributor.md#typedef-statisticst) & | [**get\_statistics**](#function-get_statistics) () const<br> |
|  [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**has\_task\_errors**](#function-has_task_errors) () const<br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**insert\_task**](#function-insert_task-12) ([**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type) task) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**insert\_task**](#function-insert_task-22) ([**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type) & task, [**double**](namespacedynampi.md#function-check_fixed_size_mpi_type) priority) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**insert\_tasks**](#function-insert_tasks) ([**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) std::vector&lt; [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; & tasks) <br> |
|  [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**is\_root\_manager**](#function-is_root_manager) () const<br> |
|  [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**remaining\_tasks\_count**](#function-remaining_tasks_count) () const<br> |
|  std::vector&lt; [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; | [**run\_tasks**](#function-run_tasks) ([**RunConfig**](structdynampi_1_1NaiveWorkDistributor_1_1RunConfig.md) config=[**RunConfig**](structdynampi_1_1NaiveWorkDistributor_1_1RunConfig.md){}) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**run\_worker**](#function-run_worker) () <br> |
|  std::vector&lt; [**TaskError**](structdynampi_1_1TaskError.md) &gt; | [**take\_task\_errors**](#function-take_task_errors) () <br> |
|   | [**~NaiveWorkDistributor**](#function-naiveworkdistributor) () <br> |




























## Public Types Documentation




### typedef StatisticsT 

```C++
using dynampi::NaiveWorkDistributor< TaskT, ResultT, Options >::StatisticsT =  std::conditional_t<statistics_mode != StatisticsMode::None, Statistics, std::monostate>;
```




<hr>
## Public Static Attributes Documentation




### variable ordered 

```C++
const bool dynampi::NaiveWorkDistributor< TaskT, ResultT, Options >::ordered;
```




<hr>
## Public Functions Documentation




### function NaiveWorkDistributor 

```C++
inline explicit dynampi::NaiveWorkDistributor::NaiveWorkDistributor (
    std::function< ResultT ( TaskT )> worker_function,
    Config runtime_config=Config {}
) 
```




<hr>



### function finalize 

```C++
inline void dynampi::NaiveWorkDistributor::finalize () 
```




<hr>



### function finish\_remaining\_tasks 

```C++
inline std::vector< ResultT > dynampi::NaiveWorkDistributor::finish_remaining_tasks () 
```




<hr>



### function get\_statistics 

```C++
inline const  StatisticsT & dynampi::NaiveWorkDistributor::get_statistics () const
```




<hr>



### function has\_task\_errors 

```C++
inline bool dynampi::NaiveWorkDistributor::has_task_errors () const
```




<hr>



### function insert\_task [1/2]

```C++
inline void dynampi::NaiveWorkDistributor::insert_task (
    TaskT task
) 
```




<hr>



### function insert\_task [2/2]

```C++
inline void dynampi::NaiveWorkDistributor::insert_task (
    const  TaskT & task,
    double priority
) 
```




<hr>



### function insert\_tasks 

```C++
inline void dynampi::NaiveWorkDistributor::insert_tasks (
    const std::vector< TaskT > & tasks
) 
```




<hr>



### function is\_root\_manager 

```C++
inline bool dynampi::NaiveWorkDistributor::is_root_manager () const
```




<hr>



### function remaining\_tasks\_count 

```C++
inline size_t dynampi::NaiveWorkDistributor::remaining_tasks_count () const
```




<hr>



### function run\_tasks 

```C++
inline std::vector< ResultT > dynampi::NaiveWorkDistributor::run_tasks (
    RunConfig config=RunConfig {}
) 
```




<hr>



### function run\_worker 

```C++
inline void dynampi::NaiveWorkDistributor::run_worker () 
```




<hr>



### function take\_task\_errors 

```C++
inline std::vector< TaskError > dynampi::NaiveWorkDistributor::take_task_errors () 
```




<hr>



### function ~NaiveWorkDistributor 

```C++
inline dynampi::NaiveWorkDistributor::~NaiveWorkDistributor () 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/naive_distributor.hpp`

