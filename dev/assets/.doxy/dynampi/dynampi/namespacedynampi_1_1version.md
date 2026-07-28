

# Namespace dynampi::version



[**Namespace List**](namespaces.md) **>** [**dynampi**](namespacedynampi.md) **>** [**version**](namespacedynampi_1_1version.md)


























## Public Attributes

| Type | Name |
| ---: | :--- |
|  constexpr int | [**major**](#variable-major)   = `DYNAMPI\_VERSION\_MAJOR`<br> |
|  constexpr int | [**minor**](#variable-minor)   = `DYNAMPI\_VERSION\_MINOR`<br> |
|  constexpr int | [**patch**](#variable-patch)   = `DYNAMPI\_VERSION\_PATCH`<br> |
|  constexpr std::string\_view | [**string**](#variable-string)   = `[**DYNAMPI\_VERSION\_STRING**](dynampi_8hpp.md#define-dynampi_version_string)`<br> |
















## Public Functions

| Type | Name |
| ---: | :--- |
|  constexpr std::string\_view | [**commit\_hash**](#function-commit_hash) () <br> |
|  constexpr std::string\_view | [**compile\_date**](#function-compile_date) () <br> |
|  constexpr bool | [**is\_at\_least**](#function-is_at_least) (int v\_major, int v\_minor, int v\_patch) <br> |




























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
constexpr bool dynampi::version::is_at_least (
    int v_major,
    int v_minor,
    int v_patch
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/dynampi.hpp`

