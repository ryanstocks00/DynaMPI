

# Namespace dynampi::version



[**Namespace List**](namespaces.md) **>** [**dynampi**](namespacedynampi.md) **>** [**version**](namespacedynampi_1_1version.md)


























## Public Attributes

| Type | Name |
| ---: | :--- |
|  [**constexpr**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**major**](#variable-major)   = `[**DYNAMPI\_VERSION\_MAJOR**](version_8hpp.md#define-dynampi_version_major)`<br> |
|  [**constexpr**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**minor**](#variable-minor)   = `[**DYNAMPI\_VERSION\_MINOR**](version_8hpp.md#define-dynampi_version_minor)`<br> |
|  [**constexpr**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**patch**](#variable-patch)   = `[**DYNAMPI\_VERSION\_PATCH**](version_8hpp.md#define-dynampi_version_patch)`<br> |
|  [**constexpr**](namespacedynampi.md#function-check_fixed_size_mpi_type) std::string\_view | [**string**](#variable-string)   = `[**DYNAMPI\_VERSION\_STRING**](version_8hpp.md#define-dynampi_version_string)`<br> |
















## Public Functions

| Type | Name |
| ---: | :--- |
|  [**constexpr**](namespacedynampi.md#function-check_fixed_size_mpi_type) std::string\_view | [**commit\_hash**](#function-commit_hash) () <br> |
|  [**constexpr**](namespacedynampi.md#function-check_fixed_size_mpi_type) std::string\_view | [**compile\_date**](#function-compile_date) () <br> |
|  [**constexpr**](namespacedynampi.md#function-check_fixed_size_mpi_type) [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**is\_at\_least**](#function-is_at_least) ([**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) v\_major, [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) v\_minor, [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) v\_patch) <br> |




























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
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/version.hpp`

