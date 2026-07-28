

# Struct dynampi::MPI\_Type&lt; std::string &gt;

**template &lt;&gt;**



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**MPI\_Type&lt; std::string &gt;**](structdynampi_1_1MPI__Type_3_01std_1_1string_01_4.md)





* `#include <mpi_types.hpp>`























## Public Static Attributes

| Type | Name |
| ---: | :--- |
|  const bool | [**resize\_required**](#variable-resize_required)   = `true`<br> |
|  const MPI\_Datatype | [**value**](#variable-value)   = `MPI\_CHAR`<br> |
















## Public Static Functions

| Type | Name |
| ---: | :--- |
|  int | [**count**](#function-count) (const std::string & str) <br> |
|  void \* | [**ptr**](#function-ptr-12) (std::string & str) noexcept<br> |
|  const void \* | [**ptr**](#function-ptr-22) (const std::string & str) noexcept<br> |
|  void | [**resize**](#function-resize) (std::string & str, int new\_size) <br> |


























## Public Static Attributes Documentation




### variable resize\_required 

```C++
const bool dynampi::MPI_Type< std::string >::resize_required;
```




<hr>



### variable value 

```C++
const MPI_Datatype dynampi::MPI_Type< std::string >::value;
```




<hr>
## Public Static Functions Documentation




### function count 

```C++
static inline int dynampi::MPI_Type< std::string >::count (
    const std::string & str
) 
```




<hr>



### function ptr [1/2]

```C++
static inline void * dynampi::MPI_Type< std::string >::ptr (
    std::string & str
) noexcept
```




<hr>



### function ptr [2/2]

```C++
static inline const void * dynampi::MPI_Type< std::string >::ptr (
    const std::string & str
) noexcept
```




<hr>



### function resize 

```C++
static inline void dynampi::MPI_Type< std::string >::resize (
    std::string & str,
    int new_size
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/mpi/mpi_types.hpp`

