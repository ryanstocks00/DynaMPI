

# Namespace dynampi::version



[**Namespace List**](namespaces.md) **>** [**dynampi**](namespacedynampi.md) **>** [**version**](namespacedynampi_1_1version.md)


























## Public Attributes

| Type | Name |
| ---: | :--- |
|  [**constexpr**](structdynampi_1_1MPI__Type.md) [**int**](structdynampi_1_1MPI__Type.md) | [**major**](#variable-major)   = `[**DYNAMPI\_VERSION\_MAJOR**](structdynampi_1_1MPI__Type.md)`<br> |
|  [**constexpr**](structdynampi_1_1MPI__Type.md) [**int**](structdynampi_1_1MPI__Type.md) | [**minor**](#variable-minor)   = `[**DYNAMPI\_VERSION\_MINOR**](structdynampi_1_1MPI__Type.md)`<br> |
|  [**constexpr**](structdynampi_1_1MPI__Type.md) [**int**](structdynampi_1_1MPI__Type.md) | [**patch**](#variable-patch)   = `[**DYNAMPI\_VERSION\_PATCH**](structdynampi_1_1MPI__Type.md)`<br> |
|  [**constexpr**](structdynampi_1_1MPI__Type.md) std::string\_view | [**string**](#variable-string)   = `[**DYNAMPI\_VERSION\_STRING**](dynampi_8hpp.md#define-dynampi_version_string)`<br> |
















## Public Functions

| Type | Name |
| ---: | :--- |
|  [**constexpr**](structdynampi_1_1MPI__Type.md) std::string\_view | [**commit\_hash**](#function-commit_hash) () <br> |
|  [**constexpr**](structdynampi_1_1MPI__Type.md) std::string\_view | [**compile\_date**](#function-compile_date) () <br> |
|  [**constexpr**](structdynampi_1_1MPI__Type.md) [**bool**](structdynampi_1_1MPI__Type.md) | [**is\_at\_least**](#function-is_at_least) ([**int**](structdynampi_1_1MPI__Type.md) v\_major, [**int**](structdynampi_1_1MPI__Type.md) v\_minor, [**int**](structdynampi_1_1MPI__Type.md) v\_patch) <br> |




























## Public Attributes Documentation




### variable major 

```C++
constexpr int dynampi::version::major;
```




<hr>



### variable minor 

```C++
constexpr int dynampi::version::minor;
```




<hr>



### variable patch 

```C++
constexpr int dynampi::version::patch;
```




<hr>



### variable string 

```C++
constexpr std::string_view dynampi::version::string;
```




<hr>
## Public Functions Documentation




### function commit\_hash 

```C++
inline constexpr std::string_view dynampi::version::commit_hash () 
```




<hr>



### function compile\_date 

```C++
inline constexpr std::string_view dynampi::version::compile_date () 
```




<hr>



### function is\_at\_least 

```C++
constexpr  bool dynampi::version::is_at_least (
    int v_major,
    int v_minor,
    int v_patch
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/dynampi.hpp`

