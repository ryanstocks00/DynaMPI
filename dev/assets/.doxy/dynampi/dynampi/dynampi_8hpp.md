

# File dynampi.hpp



[**FileList**](files.md) **>** [**dynampi**](dir_f8db417ebd5c3d89eea80c24e3fd4369.md) **>** [**dynampi.hpp**](dynampi_8hpp.md)

[Go to the source code of this file](dynampi_8hpp_source.md)



* `#include <mpi.h>`
* `#include <cassert>`
* `#include <functional>`
* `#include <optional>`
* `#include <string_view>`
* `#include <tuple>`
* `#include <vector>`
* `#include "dynampi/impl/hierarchical_distributor.hpp"`
* `#include "dynampi/impl/lockfree_distributor.hpp"`
* `#include "dynampi/impl/naive_distributor.hpp"`













## Namespaces

| Type | Name |
| ---: | :--- |
| namespace | [**dynampi**](namespacedynampi.md) <br> |
| namespace | [**version**](namespacedynampi_1_1version.md) <br> |



















































## Macros

| Type | Name |
| ---: | :--- |
| define  | [**DYNAMPI\_STR**](dynampi_8hpp.md#define-dynampi_str) (x) `[**DYNAMPI\_STR\_HELPER**](dynampi_8hpp.md#define-dynampi_str_helper)(x)`<br> |
| define  | [**DYNAMPI\_STR\_HELPER**](dynampi_8hpp.md#define-dynampi_str_helper) (x) `#x`<br> |
| define  | [**DYNAMPI\_VERSION\_STRING**](dynampi_8hpp.md#define-dynampi_version_string)  `/* multi line expression */`<br> |

## Macro Definition Documentation





### define DYNAMPI\_STR 

```C++
#define DYNAMPI_STR (
    x
) `DYNAMPI_STR_HELPER (x)`
```




<hr>



### define DYNAMPI\_STR\_HELPER 

```C++
#define DYNAMPI_STR_HELPER (
    x
) `#x`
```




<hr>



### define DYNAMPI\_VERSION\_STRING 

```C++
#define DYNAMPI_VERSION_STRING `/* multi line expression */`
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/dynampi.hpp`

