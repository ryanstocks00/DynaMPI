

# Class dynampi::MPIGroup



[**ClassList**](annotated.md) **>** [**dynampi**](namespacedynampi.md) **>** [**MPIGroup**](classdynampi_1_1MPIGroup.md)





* `#include <mpi_group.hpp>`





































## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**MPIGroup**](#function-mpigroup-13) ([**const**](structdynampi_1_1MPI__Type.md) [**MPICommunicator**](classdynampi_1_1MPICommunicator.md)&lt; Options... &gt; & comm) <br> |
|   | [**MPIGroup**](#function-mpigroup-23) ([**const**](structdynampi_1_1MPI__Type.md) [**MPIGroup**](classdynampi_1_1MPIGroup.md) & other) = delete<br> |
|   | [**MPIGroup**](#function-mpigroup-33) ([**MPIGroup**](classdynampi_1_1MPIGroup.md) && other) noexcept<br> |
|  [**MPI\_Group**](structdynampi_1_1MPI__Type.md) | [**get**](#function-get) () const<br> |
|   | [**operator MPI\_Group**](#function-operator-mpi_group) () const<br> |
|  [**MPIGroup**](classdynampi_1_1MPIGroup.md) & | [**operator=**](#function-operator) ([**const**](structdynampi_1_1MPI__Type.md) [**MPIGroup**](classdynampi_1_1MPIGroup.md) & other) = delete<br> |
|  [**MPIGroup**](classdynampi_1_1MPIGroup.md) & | [**operator=**](#function-operator_1) ([**MPIGroup**](classdynampi_1_1MPIGroup.md) && other) noexcept<br> |
|  [**int**](structdynampi_1_1MPI__Type.md) | [**rank**](#function-rank) () const<br> |
|  [**int**](structdynampi_1_1MPI__Type.md) | [**size**](#function-size) () const<br> |
|  [**int**](structdynampi_1_1MPI__Type.md) | [**translate\_rank**](#function-translate_rank) ([**int**](structdynampi_1_1MPI__Type.md) rank, [**const**](structdynampi_1_1MPI__Type.md) [**MPIGroup**](classdynampi_1_1MPIGroup.md) & to\_group) const<br> |
|  [**void**](structdynampi_1_1MPI__Type.md) | [**translate\_ranks**](#function-translate_ranks) ([**const**](structdynampi_1_1MPI__Type.md) [**MPIGroup**](classdynampi_1_1MPIGroup.md) & to\_group, [**int**](structdynampi_1_1MPI__Type.md) n, [**const**](structdynampi_1_1MPI__Type.md) [**int**](structdynampi_1_1MPI__Type.md) ranks, [**int**](structdynampi_1_1MPI__Type.md) translated\_ranks) const<br> |
|   | [**~MPIGroup**](#function-mpigroup) () <br> |


## Public Static Functions

| Type | Name |
| ---: | :--- |
|  [**int**](structdynampi_1_1MPI__Type.md) | [**contains\_rank\_in\_group**](#function-contains_rank_in_group) ([**int**](structdynampi_1_1MPI__Type.md) rank\_in\_reference, [**const**](structdynampi_1_1MPI__Type.md) [**MPIGroup**](classdynampi_1_1MPIGroup.md) & reference\_group, [**const**](structdynampi_1_1MPI__Type.md) [**MPIGroup**](classdynampi_1_1MPIGroup.md) & target\_group) <br> |


























## Public Functions Documentation




### function MPIGroup [1/3]

```C++
template<typename... Options>
inline explicit dynampi::MPIGroup::MPIGroup (
    const  MPICommunicator < Options... > & comm
) 
```




<hr>



### function MPIGroup [2/3]

```C++
dynampi::MPIGroup::MPIGroup (
    const  MPIGroup & other
) = delete
```




<hr>



### function MPIGroup [3/3]

```C++
inline dynampi::MPIGroup::MPIGroup (
    MPIGroup && other
) noexcept
```




<hr>



### function get 

```C++
inline MPI_Group dynampi::MPIGroup::get () const
```




<hr>



### function operator MPI\_Group 

```C++
inline dynampi::MPIGroup::operator MPI_Group () const
```




<hr>



### function operator= 

```C++
MPIGroup & dynampi::MPIGroup::operator= (
    const  MPIGroup & other
) = delete
```




<hr>



### function operator= 

```C++
inline MPIGroup & dynampi::MPIGroup::operator= (
    MPIGroup && other
) noexcept
```




<hr>



### function rank 

```C++
inline int dynampi::MPIGroup::rank () const
```




<hr>



### function size 

```C++
inline int dynampi::MPIGroup::size () const
```




<hr>



### function translate\_rank 

```C++
inline int dynampi::MPIGroup::translate_rank (
    int rank,
    const  MPIGroup & to_group
) const
```




<hr>



### function translate\_ranks 

```C++
inline void dynampi::MPIGroup::translate_ranks (
    const  MPIGroup & to_group,
    int n,
    const  int ranks,
    int translated_ranks
) const
```




<hr>



### function ~MPIGroup 

```C++
inline dynampi::MPIGroup::~MPIGroup () 
```




<hr>
## Public Static Functions Documentation




### function contains\_rank\_in\_group 

```C++
static inline int dynampi::MPIGroup::contains_rank_in_group (
    int rank_in_reference,
    const  MPIGroup & reference_group,
    const  MPIGroup & target_group
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/mpi/mpi_group.hpp`

