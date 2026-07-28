

# Struct dynampi::MPI\_Type&lt; std::vector&lt; T &gt;, std::enable\_if\_t&lt; has\_dynampi\_mpi\_type&lt; T &gt;::value &gt; &gt;

**template &lt;typename T&gt;**



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**MPI\_Type&lt; std::vector&lt; T &gt;, std::enable\_if\_t&lt; has\_dynampi\_mpi\_type&lt; T &gt;::value &gt; &gt;**](structdynampi_1_1MPI__Type_3_01std_1_1vector_3_01T_01_4_00_01std_1_1enable__if__t_3_01has__dynam0c05b0754f90b71498257126104ee051.md)





* `#include <mpi_types.hpp>`























## Public Static Attributes

| Type | Name |
| ---: | :--- |
|  const bool | [**resize\_required**](#variable-resize_required)   = `true`<br> |
|  const MPI\_Datatype | [**value**](#variable-value)   = `[**MPI\_Type**](structdynampi_1_1MPI__Type.md)&lt;T&gt;::value`<br> |
















## Public Static Functions

| Type | Name |
| ---: | :--- |
|  int | [**count**](#function-count) (const std::vector&lt; T &gt; & vec) <br> |
|  void \* | [**ptr**](#function-ptr-12) (std::vector&lt; T &gt; & vec) noexcept<br> |
|  const void \* | [**ptr**](#function-ptr-22) (const std::vector&lt; T &gt; & vec) noexcept<br> |
|  void | [**resize**](#function-resize) (std::vector&lt; T &gt; & vec, int new\_size) <br> |


























## Public Static Attributes Documentation




### variable resize\_required 

```C++
const bool dynampi::MPI_Type< std::vector< T >, std::enable_if_t< has_dynampi_mpi_type< T >::value > >::resize_required;
```




<hr>



### variable value 

```C++
const MPI_Datatype dynampi::MPI_Type< std::vector< T >, std::enable_if_t< has_dynampi_mpi_type< T >::value > >::value;
```




<hr>
## Public Static Functions Documentation




### function count 

```C++
static inline int dynampi::MPI_Type< std::vector< T >, std::enable_if_t< has_dynampi_mpi_type< T >::value > >::count (
    const std::vector< T > & vec
) 
```




<hr>



### function ptr [1/2]

```C++
static inline void * dynampi::MPI_Type< std::vector< T >, std::enable_if_t< has_dynampi_mpi_type< T >::value > >::ptr (
    std::vector< T > & vec
) noexcept
```




<hr>



### function ptr [2/2]

```C++
static inline const void * dynampi::MPI_Type< std::vector< T >, std::enable_if_t< has_dynampi_mpi_type< T >::value > >::ptr (
    const std::vector< T > & vec
) noexcept
```




<hr>



### function resize 

```C++
static inline void dynampi::MPI_Type< std::vector< T >, std::enable_if_t< has_dynampi_mpi_type< T >::value > >::resize (
    std::vector< T > & vec,
    int new_size
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/mpi/mpi_types.hpp`

