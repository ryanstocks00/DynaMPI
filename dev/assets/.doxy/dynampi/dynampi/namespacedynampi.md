

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
| class | [**AsyncPutLockFreeMPIWorkDistributor**](classdynampi_1_1AsyncPutLockFreeMPIWorkDistributor.md) &lt;typename TaskT, typename ResultT, Options&gt;<br> |
| class | [**BaseMPIWorkDistributor**](classdynampi_1_1BaseMPIWorkDistributor.md) &lt;typename TaskT, typename ResultT, Options&gt;<br> |
| struct | [**CommStatistics**](structdynampi_1_1CommStatistics.md) <br> |
| class | [**HierarchicalAsyncPutLockFreeMPIWorkDistributor**](classdynampi_1_1HierarchicalAsyncPutLockFreeMPIWorkDistributor.md) &lt;typename TaskT, typename ResultT, Options&gt;<br> |
| class | [**HierarchicalLockFreeMPIWorkDistributor**](classdynampi_1_1HierarchicalLockFreeMPIWorkDistributor.md) &lt;typename TaskT, typename ResultT, Options&gt;<br> |
| class | [**HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md) &lt;typename TaskT, typename ResultT, Options&gt;<br> |
| class | [**HierarchicalNonBlockingMPIWorkDistributor**](classdynampi_1_1HierarchicalNonBlockingMPIWorkDistributor.md) &lt;typename TaskT, typename ResultT, Options&gt;<br> |
| class | [**LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md) &lt;typename TaskT, typename ResultT, Options&gt;<br> |
| class | [**MPICommunicator**](classdynampi_1_1MPICommunicator.md) &lt;Options&gt;<br> |
| class | [**MPIGroup**](classdynampi_1_1MPIGroup.md) <br> |
| struct | [**MPI\_Type**](structdynampi_1_1MPI__Type.md) &lt;typename T, typename&gt;<br> |
| struct | [**MPI\_Type&lt; std::nullptr\_t &gt;**](structdynampi_1_1MPI__Type_3_01std_1_1nullptr__t_01_4.md) &lt;&gt;<br> |
| struct | [**MPI\_Type&lt; std::string &gt;**](structdynampi_1_1MPI__Type_3_01std_1_1string_01_4.md) &lt;&gt;<br> |
| struct | [**MPI\_Type&lt; std::vector&lt; T &gt;, std::enable\_if\_t&lt; has\_dynampi\_mpi\_type&lt; T &gt;::value &gt; &gt;**](structdynampi_1_1MPI__Type_3_01std_1_1vector_3_01T_01_4_00_01std_1_1enable__if__t_3_01has__dynam0c05b0754f90b71498257126104ee051.md) &lt;typename T&gt;<br> |
| class | [**MinimalLockFreeMPIWorkDistributor**](classdynampi_1_1MinimalLockFreeMPIWorkDistributor.md) &lt;typename ResultT&gt;<br> |
| class | [**NaiveMPIWorkDistributor**](classdynampi_1_1NaiveMPIWorkDistributor.md) &lt;typename TaskT, typename ResultT, Options&gt;<br> |
| class | [**Timer**](classdynampi_1_1Timer.md) <br> |
| struct | [**enable\_prioritization**](structdynampi_1_1enable__prioritization.md) <br> |
| struct | [**has\_dynampi\_mpi\_type**](structdynampi_1_1has__dynampi__mpi__type.md) &lt;typename, typename&gt;<br> |
| struct | [**has\_dynampi\_mpi\_type&lt; U, std::void\_t&lt; decltype(MPI\_Type&lt; U &gt;::value)&gt; &gt;**](structdynampi_1_1has__dynampi__mpi__type_3_01U_00_01std_1_1void__t_3_01decltype_07MPI__Type_3_01U_01_4_1_1value_08_4_01_4.md) &lt;typename U&gt;<br> |
| struct | [**prioritize\_tasks\_t**](structdynampi_1_1prioritize__tasks__t.md) <br> |
| struct | [**track\_statistics**](structdynampi_1_1track__statistics.md) &lt;Mode&gt;<br> |
| struct | [**track\_statistics\_t**](structdynampi_1_1track__statistics__t.md) <br> |


## Public Types

| Type | Name |
| ---: | :--- |
| typedef [**HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md)&lt; TaskT, ResultT, Options... &gt; | [**MPIDynamicWorkDistributor**](#typedef-mpidynamicworkdistributor)  <br> |
| enum  | [**StatisticsMode**](#enum-statisticsmode)  <br> |




















## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) (char, MPI\_CHAR) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) (std::byte, MPI\_BYTE) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) (signed char, MPI\_SIGNED\_CHAR) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) (unsigned char, MPI\_UNSIGNED\_CHAR) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) (short, MPI\_SHORT) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) (unsigned short, MPI\_UNSIGNED\_SHORT) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) (int, MPI\_INT) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) (unsigned int, MPI\_UNSIGNED) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) (long, MPI\_LONG) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) (unsigned long, MPI\_UNSIGNED\_LONG) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) (long long, MPI\_LONG\_LONG\_INT) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) (unsigned long long, MPI\_UNSIGNED\_LONG\_LONG) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) (float, MPI\_FLOAT) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) (double, MPI\_DOUBLE) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) (long double, MPI\_LONG\_DOUBLE) <br> |
|   | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](#function-dynampi_define_primitive_mpi_type) (bool, MPI\_C\_BOOL) <br> |
|  std::optional&lt; std::string &gt; | [**OptionalString**](#function-optionalstring) (Args &&... args) <br> |
|  void | [**\_DYNAMPI\_FAILBinOp**](#function-_dynampi_failbinop) (const A & a, const B & b, const std::string & a\_str, const std::string & b\_str, const std::string & nop, const std::optional&lt; std::string &gt; & message) <br> |
|  void | [**\_DYNAMPI\_FAIL\_ASSERT**](#function-_dynampi_fail_assert) (const std::string & condition\_str, const std::optional&lt; std::string &gt; & message) <br> |
|  void | [**mpi\_fail**](#function-mpi_fail) (int err, std::string\_view command) <br> |
|  std::optional&lt; std::vector&lt; ResultT &gt; &gt; | [**mpi\_manager\_worker\_distribution**](#function-mpi_manager_worker_distribution) (size\_t n\_tasks, std::function&lt; ResultT(size\_t)&gt; worker\_function, MPI\_Comm comm=MPI\_COMM\_WORLD, int manager\_rank=0) <br> |
|  std::ostream & | [**operator&lt;&lt;**](#function-operator) (std::ostream & os, const std::set&lt; T &gt; & set) <br> |
|  std::ostream & | [**operator&lt;&lt;**](#function-operator_1) (std::ostream & os, const std::vector&lt; T &gt; & vec) <br> |
|  std::ostream & | [**operator&lt;&lt;**](#function-operator_2) (std::ostream & os, const std::array&lt; T, N &gt; & arr) <br> |
|  std::ostream & | [**operator&lt;&lt;**](#function-operator_3) (std::ostream & os, const std::span&lt; T &gt; & vec) <br> |
|  std::ostream & | [**operator&lt;&lt;**](#function-operator_4) (std::ostream & os, const std::optional&lt; T &gt; & op) <br> |
|  std::ostream & | [**operator&lt;&lt;**](#function-operator_5) (std::ostream & os, const std::tuple&lt; Args... &gt; & tup) <br> |
|  std::ostream & | [**operator&lt;&lt;**](#function-operator_6) (std::ostream & os, const std::pair&lt; T, U &gt; & pair) <br> |
|  std::ostream & | [**operator&lt;&lt;**](#function-operator_7) (std::ostream & os, const std::byte & b) <br> |




























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
    signed char,
    MPI_SIGNED_CHAR
) 
```




<hr>



### function DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE 

```C++
dynampi::DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE (
    unsigned char,
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
    unsigned short,
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
    unsigned int,
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
    unsigned long,
    MPI_UNSIGNED_LONG
) 
```




<hr>



### function DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE 

```C++
dynampi::DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE (
    long long,
    MPI_LONG_LONG_INT
) 
```




<hr>



### function DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE 

```C++
dynampi::DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE (
    unsigned long long,
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
    long double,
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
    const A & a,
    const B & b,
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
    std::function< ResultT(size_t)> worker_function,
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
    const std::array< T, N > & arr
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
    const std::pair< T, U > & pair
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

