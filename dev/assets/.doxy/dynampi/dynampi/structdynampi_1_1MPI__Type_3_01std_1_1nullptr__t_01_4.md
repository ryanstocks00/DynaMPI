

# Struct dynampi::MPI\_Type&lt; std::nullptr\_t &gt;

**template &lt;&gt;**



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**MPI\_Type&lt; std::nullptr\_t &gt;**](structdynampi_1_1MPI__Type_3_01std_1_1nullptr__t_01_4.md)





* `#include <mpi_types.hpp>`























## Public Static Attributes

| Type | Name |
| ---: | :--- |
|  [**const**](structdynampi_1_1MPI__Type.md) [**bool**](structdynampi_1_1MPI__Type.md) | [**resize\_required**](#variable-resize_required)   = `[**false**](structdynampi_1_1MPI__Type.md)`<br> |
|  [**const**](structdynampi_1_1MPI__Type.md) [**MPI\_Datatype**](structdynampi_1_1MPI__Type.md) | [**value**](#variable-value)   = `[**MPI\_PACKED**](structdynampi_1_1MPI__Type.md)`<br> |
















## Public Static Functions

| Type | Name |
| ---: | :--- |
|  [**int**](structdynampi_1_1MPI__Type.md) | [**count**](#function-count) ([**const**](structdynampi_1_1MPI__Type.md) std::nullptr\_t &) noexcept<br> |
|  [**void**](structdynampi_1_1MPI__Type.md) \* | [**ptr**](#function-ptr-12) (std::nullptr\_t &) noexcept<br> |
|  [**const**](structdynampi_1_1MPI__Type.md) [**void**](structdynampi_1_1MPI__Type.md) \* | [**ptr**](#function-ptr-22) ([**const**](structdynampi_1_1MPI__Type.md) std::nullptr\_t &) noexcept<br> |
|  [**void**](structdynampi_1_1MPI__Type.md) | [**resize**](#function-resize) (std::nullptr\_t &, [**int**](structdynampi_1_1MPI__Type.md) new\_size) noexcept<br> |


























## Public Static Attributes Documentation




### variable resize\_required 

```C++
const bool dynampi::MPI_Type< std::nullptr_t >::resize_required;
```




<hr>



### variable value 

```C++
const MPI_Datatype dynampi::MPI_Type< std::nullptr_t >::value;
```




<hr>
## Public Static Functions Documentation




### function count 

```C++
static inline int dynampi::MPI_Type< std::nullptr_t >::count (
    const std::nullptr_t &
) noexcept
```




<hr>



### function ptr [1/2]

```C++
static inline void * dynampi::MPI_Type< std::nullptr_t >::ptr (
    std::nullptr_t &
) noexcept
```




<hr>



### function ptr [2/2]

```C++
static inline const  void * dynampi::MPI_Type< std::nullptr_t >::ptr (
    const std::nullptr_t &
) noexcept
```




<hr>



### function resize 

```C++
static inline void dynampi::MPI_Type< std::nullptr_t >::resize (
    std::nullptr_t &,
    int new_size
) noexcept
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/mpi/mpi_types.hpp`

