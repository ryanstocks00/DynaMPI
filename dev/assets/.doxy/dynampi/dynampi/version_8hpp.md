

# File version.hpp



[**FileList**](files.md) **>** [**dynampi**](dir_f8db417ebd5c3d89eea80c24e3fd4369.md) **>** [**version.hpp**](version_8hpp.md)

[Go to the source code of this file](version_8hpp_source.md)



* `#include <string_view>`
* `#include <tuple>`













## Namespaces

| Type | Name |
| ---: | :--- |
| namespace | [**dynampi**](namespacedynampi.md) <br> |
| namespace | [**version**](namespacedynampi_1_1version.md) <br> |



















































## Macros

| Type | Name |
| ---: | :--- |
| define  | [**DYNAMPI\_COMMIT\_HASH**](version_8hpp.md#define-dynampi_commit_hash)  `"unknown"`<br> |
| define  | [**DYNAMPI\_STR**](version_8hpp.md#define-dynampi_str) (x) `[**DYNAMPI\_STR\_HELPER**](version_8hpp.md#define-dynampi_str_helper)(x)`<br> |
| define  | [**DYNAMPI\_STR\_HELPER**](version_8hpp.md#define-dynampi_str_helper) (x) `#x`<br> |
| define  | [**DYNAMPI\_VERSION\_MAJOR**](version_8hpp.md#define-dynampi_version_major)  `0`<br> |
| define  | [**DYNAMPI\_VERSION\_MINOR**](version_8hpp.md#define-dynampi_version_minor)  `0`<br> |
| define  | [**DYNAMPI\_VERSION\_PATCH**](version_8hpp.md#define-dynampi_version_patch)  `1`<br> |
| define  | [**DYNAMPI\_VERSION\_STRING**](version_8hpp.md#define-dynampi_version_string)  `/* multi line expression */`<br> |

## Macro Definition Documentation





### define DYNAMPI\_COMMIT\_HASH 

```C++
#define DYNAMPI_COMMIT_HASH `"unknown"`
```




<hr>



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



### define DYNAMPI\_VERSION\_MAJOR 

```C++
#define DYNAMPI_VERSION_MAJOR `0`
```




<hr>



### define DYNAMPI\_VERSION\_MINOR 

```C++
#define DYNAMPI_VERSION_MINOR `0`
```




<hr>



### define DYNAMPI\_VERSION\_PATCH 

```C++
#define DYNAMPI_VERSION_PATCH `1`
```




<hr>



### define DYNAMPI\_VERSION\_STRING 

```C++
#define DYNAMPI_VERSION_STRING `"v" DYNAMPI_STR ( DYNAMPI_VERSION_MAJOR ) "." DYNAMPI_STR ( DYNAMPI_VERSION_MINOR ) "." DYNAMPI_STR ( \ DYNAMPI_VERSION_PATCH )`
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/version.hpp`

