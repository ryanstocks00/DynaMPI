

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
| class | [**BaseWorkDistributor**](classdynampi_1_1BaseWorkDistributor.md) &lt;[**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type), Options&gt;<br> |
| struct | [**CommStatistics**](structdynampi_1_1CommStatistics.md) <br> |
| class | [**HierarchicalLockFreeRMAWorkDistributor**](classdynampi_1_1HierarchicalLockFreeRMAWorkDistributor.md) &lt;[**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type), Options&gt;<br> |
| class | [**HierarchicalWorkDistributor**](classdynampi_1_1HierarchicalWorkDistributor.md) &lt;[**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type), Options&gt;<br> |
| class | [**LockFreeRMAWorkDistributor**](classdynampi_1_1LockFreeRMAWorkDistributor.md) &lt;[**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type), Options&gt;<br> |
| class | [**MPICommunicator**](classdynampi_1_1MPICommunicator.md) &lt;Options&gt;<br> |
| class | [**MPIGroup**](classdynampi_1_1MPIGroup.md) <br> |
| struct | [**MPI\_Type**](structdynampi_1_1MPI__Type.md) &lt;[**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**T**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type)&gt;<br> |
| struct | [**MPI\_Type&lt; std::nullptr\_t &gt;**](structdynampi_1_1MPI__Type_3_01std_1_1nullptr__t_01_4.md) &lt;&gt;<br> |
| struct | [**MPI\_Type&lt; std::string &gt;**](structdynampi_1_1MPI__Type_3_01std_1_1string_01_4.md) &lt;&gt;<br> |
| struct | [**MPI\_Type&lt; std::vector&lt; T &gt;, std::enable\_if\_t&lt; has\_dynampi\_mpi\_type&lt; T &gt;::value &gt; &gt;**](structdynampi_1_1MPI__Type_3_01std_1_1vector_3_01T_01_4_00_01std_1_1enable__if__t_3_01has__dynam0c05b0754f90b71498257126104ee051.md) &lt;[**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**T**](namespacedynampi.md#function-check_fixed_size_mpi_type)&gt;<br> |
| class | [**NaiveWorkDistributor**](classdynampi_1_1NaiveWorkDistributor.md) &lt;[**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type), Options&gt;<br> |
| struct | [**TaskError**](structdynampi_1_1TaskError.md) <br> |
| class | [**TaskFailure**](classdynampi_1_1TaskFailure.md) <br> |
| class | [**Timer**](classdynampi_1_1Timer.md) <br> |
| struct | [**enable\_prioritization**](structdynampi_1_1enable__prioritization.md) <br> |
| struct | [**has\_dynampi\_mpi\_type**](structdynampi_1_1has__dynampi__mpi__type.md) &lt;[**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type)&gt;<br> |
| struct | [**has\_dynampi\_mpi\_type&lt; U, std::void\_t&lt; decltype(MPI\_Type&lt; U &gt;::value)&gt; &gt;**](structdynampi_1_1has__dynampi__mpi__type_3_01U_00_01std_1_1void__t_3_01decltype_07MPI__Type_3_01U_01_4_1_1value_08_4_01_4.md) &lt;[**typename**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**U**](namespacedynampi.md#function-check_fixed_size_mpi_type)&gt;<br> |
| struct | [**prioritize\_tasks\_t**](structdynampi_1_1prioritize__tasks__t.md) <br> |
| struct | [**track\_statistics**](structdynampi_1_1track__statistics.md) &lt;Mode&gt;<br> |
| struct | [**track\_statistics\_t**](structdynampi_1_1track__statistics__t.md) <br> |


## Public Types

| Type | Name |
| ---: | :--- |
| typedef [**HierarchicalWorkDistributor**](classdynampi_1_1HierarchicalWorkDistributor.md)&lt; [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type), Options... &gt; | [**DynamicWorkDistributor**](#typedef-dynamicworkdistributor)  <br> |
| enum  | [**StatisticsMode**](#enum-statisticsmode)  <br> |




## Public Attributes

| Type | Name |
| ---: | :--- |
|  [**constexpr**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**kMaxTaskErrorMessage**](#variable-kmaxtaskerrormessage)   = `240`<br> |
















## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) ([**char**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**MPI\_CHAR**](namespacedynampi.md#function-check_fixed_size_mpi_type)) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) (std::byte, [**MPI\_BYTE**](namespacedynampi.md#function-check_fixed_size_mpi_type)) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) ([**signed**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**char**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**MPI\_SIGNED\_CHAR**](namespacedynampi.md#function-check_fixed_size_mpi_type)) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) ([**unsigned**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**char**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**MPI\_UNSIGNED\_CHAR**](namespacedynampi.md#function-check_fixed_size_mpi_type)) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) ([**short**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**MPI\_SHORT**](namespacedynampi.md#function-check_fixed_size_mpi_type)) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) ([**unsigned**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**short**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**MPI\_UNSIGNED\_SHORT**](namespacedynampi.md#function-check_fixed_size_mpi_type)) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) ([**int**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**MPI\_INT**](namespacedynampi.md#function-check_fixed_size_mpi_type)) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) ([**unsigned**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**MPI\_UNSIGNED**](namespacedynampi.md#function-check_fixed_size_mpi_type)) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) ([**long**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**MPI\_LONG**](namespacedynampi.md#function-check_fixed_size_mpi_type)) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) ([**unsigned**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**long**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**MPI\_UNSIGNED\_LONG**](namespacedynampi.md#function-check_fixed_size_mpi_type)) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) ([**long**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**long**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**MPI\_LONG\_LONG\_INT**](namespacedynampi.md#function-check_fixed_size_mpi_type)) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) ([**unsigned**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**long**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**long**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**MPI\_UNSIGNED\_LONG\_LONG**](namespacedynampi.md#function-check_fixed_size_mpi_type)) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) ([**float**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**MPI\_FLOAT**](namespacedynampi.md#function-check_fixed_size_mpi_type)) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) ([**double**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**MPI\_DOUBLE**](namespacedynampi.md#function-check_fixed_size_mpi_type)) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) ([**long**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**double**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**MPI\_LONG\_DOUBLE**](namespacedynampi.md#function-check_fixed_size_mpi_type)) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) ([**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**MPI\_C\_BOOL**](namespacedynampi.md#function-check_fixed_size_mpi_type)) <br> |
|  std::optional&lt; std::string &gt; | [**OptionalString**](#function-optionalstring) ([**Args**](namespacedynampi.md#function-check_fixed_size_mpi_type) &&... args) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**\_DYNAMPI\_FAILBinOp**](#function-_dynampi_failbinop) ([**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**A**](namespacedynampi.md#function-check_fixed_size_mpi_type) & a, [**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**B**](namespacedynampi.md#function-check_fixed_size_mpi_type) & b, [**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) std::string & a\_str, [**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) std::string & b\_str, [**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) std::string & nop, [**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) std::optional&lt; std::string &gt; & message) <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**\_DYNAMPI\_FAIL\_ASSERT**](#function-_dynampi_fail_assert) ([**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) std::string & condition\_str, [**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) std::optional&lt; std::string &gt; & message) <br> |
|  void | [**check\_fixed\_size\_mpi\_type**](#function-check_fixed_size_mpi_type) (const char \* type\_role, const char \* distributor\_name) <br> |
|  [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**mpi\_elements\_per\_value**](#function-mpi_elements_per_value) () <br> |
|  [**void**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**mpi\_fail**](#function-mpi_fail) ([**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) err, std::string\_view command) <br> |
|  std::optional&lt; std::vector&lt; [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; &gt; | [**mpi\_manager\_worker\_distribution**](#function-mpi_manager_worker_distribution) ([**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) n\_tasks, std::function&lt; [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type)([**size\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type))&gt; worker\_function, [**MPI\_Comm**](namespacedynampi.md#function-check_fixed_size_mpi_type) comm=[**MPI\_COMM\_WORLD**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) manager\_rank=0) <br> |
|  std::ostream & | [**operator&lt;&lt;**](#function-operator) (std::ostream & os, [**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) std::set&lt; [**T**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; & set) <br> |
|  std::ostream & | [**operator&lt;&lt;**](#function-operator_1) (std::ostream & os, [**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) std::vector&lt; [**T**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; & vec) <br> |
|  std::ostream & | [**operator&lt;&lt;**](#function-operator_2) (std::ostream & os, [**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) std::array&lt; [**T**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**N**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; & arr) <br> |
|  std::ostream & | [**operator&lt;&lt;**](#function-operator_3) (std::ostream & os, [**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) std::span&lt; [**T**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; & vec) <br> |
|  std::ostream & | [**operator&lt;&lt;**](#function-operator_4) (std::ostream & os, [**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) std::optional&lt; [**T**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; & op) <br> |
|  std::ostream & | [**operator&lt;&lt;**](#function-operator_5) (std::ostream & os, [**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) std::tuple&lt; Args... &gt; & tup) <br> |
|  std::ostream & | [**operator&lt;&lt;**](#function-operator_6) (std::ostream & os, [**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) std::pair&lt; [**T**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**U**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; & pair) <br> |
|  std::ostream & | [**operator&lt;&lt;**](#function-operator_7) (std::ostream & os, [**const**](namespacedynampi.md#function-check_fixed_size_mpi_type) std::byte & b) <br> |




























## Public Types Documentation




### typedef DynamicWorkDistributor 

```C++
using dynampi::DynamicWorkDistributor = typedef HierarchicalWorkDistributor<TaskT, ResultT, Options...>;
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
## Public Attributes Documentation




### variable kMaxTaskErrorMessage 

```C++
constexpr size_t dynampi::kMaxTaskErrorMessage;
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



### function check\_fixed\_size\_mpi\_type 

```C++
template<typename T>
inline void dynampi::check_fixed_size_mpi_type (
    const char * type_role,
    const char * distributor_name
) 
```




<hr>



### function mpi\_elements\_per\_value 

```C++
template<typename T>
inline int dynampi::mpi_elements_per_value () 
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

