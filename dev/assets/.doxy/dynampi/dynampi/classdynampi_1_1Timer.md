

# Class dynampi::Timer



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**Timer**](classdynampi_1_1Timer.md)





* `#include <timer.hpp>`

















## Public Types

| Type | Name |
| ---: | :--- |
| enum  | [**AutoStart**](#enum-autostart)  <br> |




















## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**Timer**](#function-timer) ([**AutoStart**](classdynampi_1_1Timer.md#enum-autostart) auto\_start=AutoStart::Yes) <br> |
|  std::chrono::duration&lt; [**double**](structdynampi_1_1MPI__Type.md) &gt; | [**elapsed**](#function-elapsed) () const<br> |
|  [**void**](structdynampi_1_1MPI__Type.md) | [**reset**](#function-reset) ([**AutoStart**](classdynampi_1_1Timer.md#enum-autostart) auto\_start=AutoStart::Yes) <br> |
|  [**void**](structdynampi_1_1MPI__Type.md) | [**start**](#function-start) () <br> |
|  std::chrono::duration&lt; [**double**](structdynampi_1_1MPI__Type.md) &gt; | [**stop**](#function-stop) () <br> |




























## Public Types Documentation




### enum AutoStart 

```C++
enum dynampi::Timer::AutoStart {
    Yes,
    No
};
```




<hr>
## Public Functions Documentation




### function Timer 

```C++
inline dynampi::Timer::Timer (
    AutoStart auto_start=AutoStart::Yes
) 
```




<hr>



### function elapsed 

```C++
inline std::chrono::duration< double > dynampi::Timer::elapsed () const
```




<hr>



### function reset 

```C++
inline void dynampi::Timer::reset (
    AutoStart auto_start=AutoStart::Yes
) 
```




<hr>



### function start 

```C++
inline void dynampi::Timer::start () 
```




<hr>



### function stop 

```C++
inline std::chrono::duration< double > dynampi::Timer::stop () 
```




<hr>## Friends Documentation





### friend operator&lt;&lt; 

```C++
inline std::ostream & dynampi::Timer::operator<< (
    std::ostream & os,
    const  Timer & timer
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/utilities/timer.hpp`

