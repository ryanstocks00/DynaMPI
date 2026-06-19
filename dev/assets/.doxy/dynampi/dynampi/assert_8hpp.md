

# File assert.hpp



[**FileList**](files.md) **>** [**dynampi**](dir_f8db417ebd5c3d89eea80c24e3fd4369.md) **>** [**utilities**](dir_23e51883c93568b92bc8806003dcc116.md) **>** [**assert.hpp**](assert_8hpp.md)

[Go to the source code of this file](assert_8hpp_source.md)



* `#include <mpi.h>`
* `#include <exception>`
* `#include <iostream>`
* `#include <optional>`
* `#include <sstream>`
* `#include <stdexcept>`
* `#include <string>`
* `#include <string_view>`
* `#include "printing.hpp"`













## Namespaces

| Type | Name |
| ---: | :--- |
| namespace | [**dynampi**](namespacedynampi.md) <br> |



















































## Macros

| Type | Name |
| ---: | :--- |
| define  | [**DYNAMPI\_ASSERT**](assert_8hpp.md#define-dynampi_assert) (condition, ...) `/* multi line expression */`<br> |
| define  | [**DYNAMPI\_ASSERT\_BIN\_OP**](assert_8hpp.md#define-dynampi_assert_bin_op) (a, b, op, nop, ...) `/* multi line expression */`<br> |
| define  | [**DYNAMPI\_ASSERT\_EQ**](assert_8hpp.md#define-dynampi_assert_eq) (expr, val, ...) `[**DYNAMPI\_ASSERT\_BIN\_OP**](assert_8hpp.md#define-dynampi_assert_bin_op)(expr, val, ==, !=, \_\_VA\_ARGS\_\_)`<br> |
| define  | [**DYNAMPI\_ASSERT\_GE**](assert_8hpp.md#define-dynampi_assert_ge) (expr, val, ...) `[**DYNAMPI\_ASSERT\_BIN\_OP**](assert_8hpp.md#define-dynampi_assert_bin_op)(expr, val, &gt;=, &lt;, \_\_VA\_ARGS\_\_)`<br> |
| define  | [**DYNAMPI\_ASSERT\_GT**](assert_8hpp.md#define-dynampi_assert_gt) (expr, val, ...) `[**DYNAMPI\_ASSERT\_BIN\_OP**](assert_8hpp.md#define-dynampi_assert_bin_op)(expr, val, &gt;, &lt;=, \_\_VA\_ARGS\_\_)`<br> |
| define  | [**DYNAMPI\_ASSERT\_LE**](assert_8hpp.md#define-dynampi_assert_le) (expr, val, ...) `[**DYNAMPI\_ASSERT\_BIN\_OP**](assert_8hpp.md#define-dynampi_assert_bin_op)(expr, val, &lt;=, &gt;, \_\_VA\_ARGS\_\_)`<br> |
| define  | [**DYNAMPI\_ASSERT\_LT**](assert_8hpp.md#define-dynampi_assert_lt) (expr, val, ...) `[**DYNAMPI\_ASSERT\_BIN\_OP**](assert_8hpp.md#define-dynampi_assert_bin_op)(expr, val, &lt;, &gt;=, \_\_VA\_ARGS\_\_)`<br> |
| define  | [**DYNAMPI\_ASSERT\_NE**](assert_8hpp.md#define-dynampi_assert_ne) (expr, val, ...) `[**DYNAMPI\_ASSERT\_BIN\_OP**](assert_8hpp.md#define-dynampi_assert_bin_op)(expr, val, !=, ==, \_\_VA\_ARGS\_\_)`<br> |
| define  | [**DYNAMPI\_FAIL**](assert_8hpp.md#define-dynampi_fail) (...) `/* multi line expression */`<br> |
| define  | [**DYNAMPI\_HAS\_BUILTIN**](assert_8hpp.md#define-dynampi_has_builtin) (x) `\_\_has\_builtin(x)`<br> |
| define  | [**DYNAMPI\_UNIMPLEMENTED**](assert_8hpp.md#define-dynampi_unimplemented) (...) `[**DYNAMPI\_FAIL**](assert_8hpp.md#define-dynampi_fail)("DYNAMPI\_UNIMPLEMENTED")`<br> |
| define  | [**DYNAMPI\_UNREACHABLE**](assert_8hpp.md#define-dynampi_unreachable) () `\_\_builtin\_unreachable()`<br> |

## Macro Definition Documentation





### define DYNAMPI\_ASSERT 

```C++
#define DYNAMPI_ASSERT (
    condition,
    ...
) `/* multi line expression */`
```




<hr>



### define DYNAMPI\_ASSERT\_BIN\_OP 

```C++
#define DYNAMPI_ASSERT_BIN_OP (
    a,
    b,
    op,
    nop,
    ...
) `/* multi line expression */`
```




<hr>



### define DYNAMPI\_ASSERT\_EQ 

```C++
#define DYNAMPI_ASSERT_EQ (
    expr,
    val,
    ...
) `DYNAMPI_ASSERT_BIN_OP (expr, val, ==, !=, __VA_ARGS__)`
```




<hr>



### define DYNAMPI\_ASSERT\_GE 

```C++
#define DYNAMPI_ASSERT_GE (
    expr,
    val,
    ...
) `DYNAMPI_ASSERT_BIN_OP (expr, val, >=, <, __VA_ARGS__)`
```




<hr>



### define DYNAMPI\_ASSERT\_GT 

```C++
#define DYNAMPI_ASSERT_GT (
    expr,
    val,
    ...
) `DYNAMPI_ASSERT_BIN_OP (expr, val, >, <=, __VA_ARGS__)`
```




<hr>



### define DYNAMPI\_ASSERT\_LE 

```C++
#define DYNAMPI_ASSERT_LE (
    expr,
    val,
    ...
) `DYNAMPI_ASSERT_BIN_OP (expr, val, <=, >, __VA_ARGS__)`
```




<hr>



### define DYNAMPI\_ASSERT\_LT 

```C++
#define DYNAMPI_ASSERT_LT (
    expr,
    val,
    ...
) `DYNAMPI_ASSERT_BIN_OP (expr, val, <, >=, __VA_ARGS__)`
```




<hr>



### define DYNAMPI\_ASSERT\_NE 

```C++
#define DYNAMPI_ASSERT_NE (
    expr,
    val,
    ...
) `DYNAMPI_ASSERT_BIN_OP (expr, val, !=, ==, __VA_ARGS__)`
```




<hr>



### define DYNAMPI\_FAIL 

```C++
#define DYNAMPI_FAIL (
    ...
) `DYNAMPI_ASSERT (false, __VA_ARGS__); \ DYNAMPI_UNREACHABLE ()`
```




<hr>



### define DYNAMPI\_HAS\_BUILTIN 

```C++
#define DYNAMPI_HAS_BUILTIN (
    x
) `__has_builtin(x)`
```




<hr>



### define DYNAMPI\_UNIMPLEMENTED 

```C++
#define DYNAMPI_UNIMPLEMENTED (
    ...
) `DYNAMPI_FAIL ("DYNAMPI_UNIMPLEMENTED")`
```




<hr>



### define DYNAMPI\_UNREACHABLE 

```C++
#define DYNAMPI_UNREACHABLE (
    
) `__builtin_unreachable()`
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/utilities/assert.hpp`

