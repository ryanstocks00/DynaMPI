

# File mpi\_types.hpp



[**FileList**](files.md) **>** [**dynampi**](dir_f8db417ebd5c3d89eea80c24e3fd4369.md) **>** [**mpi**](dir_70f9944cf42b7c83e40875b744e30ff7.md) **>** [**mpi\_types.hpp**](mpi__types_8hpp.md)

[Go to the source code of this file](mpi__types_8hpp_source.md)



* `#include <mpi.h>`
* `#include <cassert>`
* `#include <cstddef>`
* `#include <string>`
* `#include <type_traits>`
* `#include <vector>`













## Namespaces

| Type | Name |
| ---: | :--- |
| namespace | [**dynampi**](namespacedynampi.md) <br> |


## Classes

| Type | Name |
| ---: | :--- |
| struct | [**MPI\_Type**](structdynampi_1_1MPI__Type.md) &lt;[**typename**](structdynampi_1_1MPI__Type.md) [**T**](structdynampi_1_1MPI__Type.md), [**typename**](structdynampi_1_1MPI__Type.md)&gt;<br> |
| struct | [**MPI\_Type&lt; std::nullptr\_t &gt;**](structdynampi_1_1MPI__Type_3_01std_1_1nullptr__t_01_4.md) &lt;&gt;<br> |
| struct | [**MPI\_Type&lt; std::string &gt;**](structdynampi_1_1MPI__Type_3_01std_1_1string_01_4.md) &lt;&gt;<br> |
| struct | [**MPI\_Type&lt; std::vector&lt; T &gt;, std::enable\_if\_t&lt; has\_dynampi\_mpi\_type&lt; T &gt;::value &gt; &gt;**](structdynampi_1_1MPI__Type_3_01std_1_1vector_3_01T_01_4_00_01std_1_1enable__if__t_3_01has__dynam0c05b0754f90b71498257126104ee051.md) &lt;[**typename**](structdynampi_1_1MPI__Type.md) [**T**](structdynampi_1_1MPI__Type.md)&gt;<br> |
| struct | [**has\_dynampi\_mpi\_type**](structdynampi_1_1has__dynampi__mpi__type.md) &lt;[**typename**](structdynampi_1_1MPI__Type.md), [**typename**](structdynampi_1_1MPI__Type.md)&gt;<br> |
| struct | [**has\_dynampi\_mpi\_type&lt; U, std::void\_t&lt; decltype(MPI\_Type&lt; U &gt;::value)&gt; &gt;**](structdynampi_1_1has__dynampi__mpi__type_3_01U_00_01std_1_1void__t_3_01decltype_07MPI__Type_3_01U_01_4_1_1value_08_4_01_4.md) &lt;[**typename**](structdynampi_1_1MPI__Type.md) [**U**](structdynampi_1_1MPI__Type.md)&gt;<br> |

















































## Macros

| Type | Name |
| ---: | :--- |
| define  | [**DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE**](mpi__types_8hpp.md#define-dynampi_define_primitive_mpi_type) (type, mpi\_type) `/* multi line expression */`<br> |

## Macro Definition Documentation





### define DYNAMPI\_DEFINE\_PRIMITIVE\_MPI\_TYPE 

```C++
#define DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE (
    type,
    mpi_type
) `/* multi line expression */`
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/mpi/mpi_types.hpp`

