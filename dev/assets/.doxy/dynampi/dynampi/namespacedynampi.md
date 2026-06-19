

# Namespace dynampi



[**Namespace List**](namespaces.md) **>** [**dynampi**](namespacedynampi.md)


















## Namespaces

| Type | Name |
| ---: | :--- |
| namespace | [**detail**](namespacedynampi_1_1detail.md) <br> |
| namespace | [**version**](namespacedynampi_1_1version.md) <br> |


## Classes

| Type | Name |
| ---: | :--- |
| class | [**BaseMPIWorkDistributor**](classdynampi_1_1BaseMPIWorkDistributor.md) &lt;[**typename**](structdynampi_1_1MPI__Type.md) [**TaskT**](structdynampi_1_1MPI__Type.md), [**typename**](structdynampi_1_1MPI__Type.md) [**ResultT**](structdynampi_1_1MPI__Type.md), Options&gt;<br> |
| struct | [**CommStatistics**](structdynampi_1_1CommStatistics.md) <br> |
| class | [**HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md) &lt;[**typename**](structdynampi_1_1MPI__Type.md) [**TaskT**](structdynampi_1_1MPI__Type.md), [**typename**](structdynampi_1_1MPI__Type.md) [**ResultT**](structdynampi_1_1MPI__Type.md), Options&gt;<br> |
| class | [**LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md) &lt;[**typename**](structdynampi_1_1MPI__Type.md) [**TaskT**](structdynampi_1_1MPI__Type.md), [**typename**](structdynampi_1_1MPI__Type.md) [**ResultT**](structdynampi_1_1MPI__Type.md), Options&gt;<br> |
| class | [**MPICommunicator**](classdynampi_1_1MPICommunicator.md) &lt;Options&gt;<br> |
| class | [**MPIGroup**](classdynampi_1_1MPIGroup.md) <br> |
| struct | [**MPI\_Type**](structdynampi_1_1MPI__Type.md) &lt;[**typename**](structdynampi_1_1MPI__Type.md) [**T**](structdynampi_1_1MPI__Type.md), [**typename**](structdynampi_1_1MPI__Type.md)&gt;<br> |
| struct | [**MPI\_Type&lt; std::nullptr\_t &gt;**](structdynampi_1_1MPI__Type_3_01std_1_1nullptr__t_01_4.md) &lt;&gt;<br> |
| struct | [**MPI\_Type&lt; std::string &gt;**](structdynampi_1_1MPI__Type_3_01std_1_1string_01_4.md) &lt;&gt;<br> |
| struct | [**MPI\_Type&lt; std::vector&lt; T &gt;, std::enable\_if\_t&lt; has\_dynampi\_mpi\_type&lt; T &gt;::value &gt; &gt;**](structdynampi_1_1MPI__Type_3_01std_1_1vector_3_01T_01_4_00_01std_1_1enable__if__t_3_01has__dynam0c05b0754f90b71498257126104ee051.md) &lt;[**typename**](structdynampi_1_1MPI__Type.md) [**T**](structdynampi_1_1MPI__Type.md)&gt;<br> |
| class | [**MinimalLockFreeMPIWorkDistributor**](classdynampi_1_1MinimalLockFreeMPIWorkDistributor.md) &lt;[**typename**](structdynampi_1_1MPI__Type.md) [**ResultT**](structdynampi_1_1MPI__Type.md)&gt;<br> |
| class | [**NaiveMPIWorkDistributor**](classdynampi_1_1NaiveMPIWorkDistributor.md) &lt;[**typename**](structdynampi_1_1MPI__Type.md) [**TaskT**](structdynampi_1_1MPI__Type.md), [**typename**](structdynampi_1_1MPI__Type.md) [**ResultT**](structdynampi_1_1MPI__Type.md), Options&gt;<br> |
| class | [**Timer**](classdynampi_1_1Timer.md) <br> |
| struct | [**enable\_prioritization**](structdynampi_1_1enable__prioritization.md) <br> |
| struct | [**has\_dynampi\_mpi\_type**](structdynampi_1_1has__dynampi__mpi__type.md) &lt;[**typename**](structdynampi_1_1MPI__Type.md), [**typename**](structdynampi_1_1MPI__Type.md)&gt;<br> |
| struct | [**has\_dynampi\_mpi\_type&lt; U, std::void\_t&lt; decltype(MPI\_Type&lt; U &gt;::value)&gt; &gt;**](structdynampi_1_1has__dynampi__mpi__type_3_01U_00_01std_1_1void__t_3_01decltype_07MPI__Type_3_01U_01_4_1_1value_08_4_01_4.md) &lt;[**typename**](structdynampi_1_1MPI__Type.md) [**U**](structdynampi_1_1MPI__Type.md)&gt;<br> |
| struct | [**prioritize\_tasks\_t**](structdynampi_1_1prioritize__tasks__t.md) <br> |
| struct | [**track\_statistics**](structdynampi_1_1track__statistics.md) &lt;Mode&gt;<br> |
| struct | [**track\_statistics\_t**](structdynampi_1_1track__statistics__t.md) <br> |


## Public Types

| Type | Name |
| ---: | :--- |
| typedef [**HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md)&lt; [**TaskT**](structdynampi_1_1MPI__Type.md), [**ResultT**](structdynampi_1_1MPI__Type.md), Options... &gt; | [**MPIDynamicWorkDistributor**](#typedef-mpidynamicworkdistributor)  <br> |
| enum  | [**StatisticsMode**](#enum-statisticsmode)  <br> |




















## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) ([**char**](structdynampi_1_1MPI__Type.md), [**MPI\_CHAR**](structdynampi_1_1MPI__Type.md)) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) (std::byte, [**MPI\_BYTE**](structdynampi_1_1MPI__Type.md)) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) ([**signed**](structdynampi_1_1MPI__Type.md) [**char**](structdynampi_1_1MPI__Type.md), [**MPI\_SIGNED\_CHAR**](structdynampi_1_1MPI__Type.md)) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) ([**unsigned**](structdynampi_1_1MPI__Type.md) [**char**](structdynampi_1_1MPI__Type.md), [**MPI\_UNSIGNED\_CHAR**](structdynampi_1_1MPI__Type.md)) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) ([**short**](structdynampi_1_1MPI__Type.md), [**MPI\_SHORT**](structdynampi_1_1MPI__Type.md)) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) ([**unsigned**](structdynampi_1_1MPI__Type.md) [**short**](structdynampi_1_1MPI__Type.md), [**MPI\_UNSIGNED\_SHORT**](structdynampi_1_1MPI__Type.md)) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) ([**int**](structdynampi_1_1MPI__Type.md), [**MPI\_INT**](structdynampi_1_1MPI__Type.md)) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) ([**unsigned**](structdynampi_1_1MPI__Type.md) [**int**](structdynampi_1_1MPI__Type.md), [**MPI\_UNSIGNED**](structdynampi_1_1MPI__Type.md)) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) ([**long**](structdynampi_1_1MPI__Type.md), [**MPI\_LONG**](structdynampi_1_1MPI__Type.md)) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) ([**unsigned**](structdynampi_1_1MPI__Type.md) [**long**](structdynampi_1_1MPI__Type.md), [**MPI\_UNSIGNED\_LONG**](structdynampi_1_1MPI__Type.md)) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) ([**long**](structdynampi_1_1MPI__Type.md) [**long**](structdynampi_1_1MPI__Type.md), [**MPI\_LONG\_LONG\_INT**](structdynampi_1_1MPI__Type.md)) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) ([**unsigned**](structdynampi_1_1MPI__Type.md) [**long**](structdynampi_1_1MPI__Type.md) [**long**](structdynampi_1_1MPI__Type.md), [**MPI\_UNSIGNED\_LONG\_LONG**](structdynampi_1_1MPI__Type.md)) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) ([**float**](structdynampi_1_1MPI__Type.md), [**MPI\_FLOAT**](structdynampi_1_1MPI__Type.md)) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) ([**double**](structdynampi_1_1MPI__Type.md), [**MPI\_DOUBLE**](structdynampi_1_1MPI__Type.md)) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) ([**long**](structdynampi_1_1MPI__Type.md) [**double**](structdynampi_1_1MPI__Type.md), [**MPI\_LONG\_DOUBLE**](structdynampi_1_1MPI__Type.md)) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) ([**bool**](structdynampi_1_1MPI__Type.md), [**MPI\_C\_BOOL**](structdynampi_1_1MPI__Type.md)) <br> |
|  std::optional&lt; std::string &gt; | [**OptionalString**](#function-optionalstring) ([**Args**](structdynampi_1_1MPI__Type.md) &&... args) <br> |
|  [**void**](structdynampi_1_1MPI__Type.md) | [**\_DYNAMPI\_FAILBinOp**](#function-_dynampi_failbinop) ([**const**](structdynampi_1_1MPI__Type.md) [**A**](structdynampi_1_1MPI__Type.md) & a, [**const**](structdynampi_1_1MPI__Type.md) [**B**](structdynampi_1_1MPI__Type.md) & b, [**const**](structdynampi_1_1MPI__Type.md) std::string & a\_str, [**const**](structdynampi_1_1MPI__Type.md) std::string & b\_str, [**const**](structdynampi_1_1MPI__Type.md) std::string & nop, [**const**](structdynampi_1_1MPI__Type.md) std::optional&lt; std::string &gt; & message) <br> |
|  [**void**](structdynampi_1_1MPI__Type.md) | [**\_DYNAMPI\_FAIL\_ASSERT**](#function-_dynampi_fail_assert) ([**const**](structdynampi_1_1MPI__Type.md) std::string & condition\_str, [**const**](structdynampi_1_1MPI__Type.md) std::optional&lt; std::string &gt; & message) <br> |
|  [**void**](structdynampi_1_1MPI__Type.md) | [**mpi\_fail**](#function-mpi_fail) ([**int**](structdynampi_1_1MPI__Type.md) err, std::string\_view command) <br> |
|  std::optional&lt; std::vector&lt; [**ResultT**](structdynampi_1_1MPI__Type.md) &gt; &gt; | [**mpi\_manager\_worker\_distribution**](#function-mpi_manager_worker_distribution) ([**size\_t**](structdynampi_1_1MPI__Type.md) n\_tasks, std::function&lt; [**ResultT**](structdynampi_1_1MPI__Type.md)([**size\_t**](structdynampi_1_1MPI__Type.md))&gt; worker\_function, [**MPI\_Comm**](structdynampi_1_1MPI__Type.md) comm=[**MPI\_COMM\_WORLD**](structdynampi_1_1MPI__Type.md), [**int**](structdynampi_1_1MPI__Type.md) manager\_rank=0) <br> |
|  std::ostream & | [**operator&lt;&lt;**](#function-operator) (std::ostream & os, [**const**](structdynampi_1_1MPI__Type.md) std::set&lt; [**T**](structdynampi_1_1MPI__Type.md) &gt; & set) <br> |
|  std::ostream & | [**operator&lt;&lt;**](#function-operator_1) (std::ostream & os, [**const**](structdynampi_1_1MPI__Type.md) std::vector&lt; [**T**](structdynampi_1_1MPI__Type.md) &gt; & vec) <br> |
|  std::ostream & | [**operator&lt;&lt;**](#function-operator_2) (std::ostream & os, [**const**](structdynampi_1_1MPI__Type.md) std::array&lt; [**T**](structdynampi_1_1MPI__Type.md), [**N**](structdynampi_1_1MPI__Type.md) &gt; & arr) <br> |
|  std::ostream & | [**operator&lt;&lt;**](#function-operator_3) (std::ostream & os, [**const**](structdynampi_1_1MPI__Type.md) std::span&lt; [**T**](structdynampi_1_1MPI__Type.md) &gt; & vec) <br> |
|  std::ostream & | [**operator&lt;&lt;**](#function-operator_4) (std::ostream & os, [**const**](structdynampi_1_1MPI__Type.md) std::optional&lt; [**T**](structdynampi_1_1MPI__Type.md) &gt; & op) <br> |
|  std::ostream & | [**operator&lt;&lt;**](#function-operator_5) (std::ostream & os, [**const**](structdynampi_1_1MPI__Type.md) std::tuple&lt; Args... &gt; & tup) <br> |
|  std::ostream & | [**operator&lt;&lt;**](#function-operator_6) (std::ostream & os, [**const**](structdynampi_1_1MPI__Type.md) std::pair&lt; [**T**](structdynampi_1_1MPI__Type.md), [**U**](structdynampi_1_1MPI__Type.md) &gt; & pair) <br> |
|  std::ostream & | [**operator&lt;&lt;**](#function-operator_7) (std::ostream & os, [**const**](structdynampi_1_1MPI__Type.md) std::byte & b) <br> |




























## Public Types Documentation




### typedef MPIDynamicWorkDistributor 

```C++
using dynampi::MPIDynamicWorkDistributor = typedef HierarchicalMPIWorkDistributor<TaskT, ResultT, Options...>;
```




<hr>



### enum StatisticsMode 

```C++
enum dynampi::StatisticsMode {
    None,
    Aggregated,
    Detailed
};
```




<hr>
## Public Functions Documentation




### function DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE 

```C++
dynampi::DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE (
    char,
    MPI_CHAR
) 
```




<hr>



### function DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE 

```C++
dynampi::DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE (
    std::byte,
    MPI_BYTE
) 
```




<hr>



### function DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE 

```C++
dynampi::DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE (
    signed  char,
    MPI_SIGNED_CHAR
) 
```




<hr>



### function DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE 

```C++
dynampi::DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE (
    unsigned  char,
    MPI_UNSIGNED_CHAR
) 
```




<hr>



### function DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE 

```C++
dynampi::DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE (
    short,
    MPI_SHORT
) 
```




<hr>



### function DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE 

```C++
dynampi::DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE (
    unsigned  short,
    MPI_UNSIGNED_SHORT
) 
```




<hr>



### function DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE 

```C++
dynampi::DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE (
    int,
    MPI_INT
) 
```




<hr>



### function DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE 

```C++
dynampi::DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE (
    unsigned  int,
    MPI_UNSIGNED
) 
```




<hr>



### function DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE 

```C++
dynampi::DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE (
    long,
    MPI_LONG
) 
```




<hr>



### function DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE 

```C++
dynampi::DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE (
    unsigned  long,
    MPI_UNSIGNED_LONG
) 
```




<hr>



### function DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE 

```C++
dynampi::DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE (
    long  long,
    MPI_LONG_LONG_INT
) 
```




<hr>



### function DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE 

```C++
dynampi::DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE (
    unsigned  long  long,
    MPI_UNSIGNED_LONG_LONG
) 
```




<hr>



### function DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE 

```C++
dynampi::DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE (
    float,
    MPI_FLOAT
) 
```




<hr>



### function DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE 

```C++
dynampi::DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE (
    double,
    MPI_DOUBLE
) 
```




<hr>



### function DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE 

```C++
dynampi::DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE (
    long  double,
    MPI_LONG_DOUBLE
) 
```




<hr>



### function DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE 

```C++
dynampi::DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE (
    bool,
    MPI_C_BOOL
) 
```




<hr>



### function OptionalString 

```C++
template<typename... Args>
std::optional< std::string > dynampi::OptionalString (
    Args &&... args
) 
```




<hr>



### function \_DYNAMPI\_FAILBinOp 

```C++
template<typename A, typename B>
inline void dynampi::_DYNAMPI_FAILBinOp (
    const  A & a,
    const  B & b,
    const std::string & a_str,
    const std::string & b_str,
    const std::string & nop,
    const std::optional< std::string > & message
) 
```




<hr>



### function \_DYNAMPI\_FAIL\_ASSERT 

```C++
inline void dynampi::_DYNAMPI_FAIL_ASSERT (
    const std::string & condition_str,
    const std::optional< std::string > & message
) 
```




<hr>



### function mpi\_fail 

```C++
inline void dynampi::mpi_fail (
    int err,
    std::string_view command
) 
```




<hr>



### function mpi\_manager\_worker\_distribution 

```C++
template<typename ResultT, template< typename, typename, typename... > typename T>
std::optional< std::vector< ResultT > > dynampi::mpi_manager_worker_distribution (
    size_t n_tasks,
    std::function< ResultT ( size_t )> worker_function,
    MPI_Comm comm=MPI_COMM_WORLD,
    int manager_rank=0
) 
```




<hr>



### function operator&lt;&lt; 

```C++
template<typename T>
inline std::ostream & dynampi::operator<< (
    std::ostream & os,
    const std::set< T > & set
) 
```




<hr>



### function operator&lt;&lt; 

```C++
template<typename T>
inline std::ostream & dynampi::operator<< (
    std::ostream & os,
    const std::vector< T > & vec
) 
```




<hr>



### function operator&lt;&lt; 

```C++
template<typename T, std::size_t N>
inline std::ostream & dynampi::operator<< (
    std::ostream & os,
    const std::array< T , N > & arr
) 
```




<hr>



### function operator&lt;&lt; 

```C++
template<typename T>
inline std::ostream & dynampi::operator<< (
    std::ostream & os,
    const std::span< T > & vec
) 
```




<hr>



### function operator&lt;&lt; 

```C++
template<typename T>
inline std::ostream & dynampi::operator<< (
    std::ostream & os,
    const std::optional< T > & op
) 
```




<hr>



### function operator&lt;&lt; 

```C++
template<typename... Args>
inline std::ostream & dynampi::operator<< (
    std::ostream & os,
    const std::tuple< Args... > & tup
) 
```




<hr>



### function operator&lt;&lt; 

```C++
template<typename T, typename U>
inline std::ostream & dynampi::operator<< (
    std::ostream & os,
    const std::pair< T , U > & pair
) 
```




<hr>



### function operator&lt;&lt; 

```C++
inline std::ostream & dynampi::operator<< (
    std::ostream & os,
    const std::byte & b
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/dynampi.hpp`

