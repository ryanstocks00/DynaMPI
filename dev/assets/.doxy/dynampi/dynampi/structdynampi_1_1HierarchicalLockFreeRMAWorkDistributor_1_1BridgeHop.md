

# Struct dynampi::HierarchicalLockFreeRMAWorkDistributor::BridgeHop



[**ClassList**](annotated.md) **>** [**BridgeHop**](structdynampi_1_1HierarchicalLockFreeRMAWorkDistributor_1_1BridgeHop.md)


























## Public Attributes

| Type | Name |
| ---: | :--- |
|  [**detail::LockFreeRMALevel**](classdynampi_1_1detail_1_1LockFreeRMALevel.md)&lt; [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; \* | [**child**](#variable-child)  <br> |
|  [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**finish\_marked**](#variable-finish_marked)   = `[**false**](namespacedynampi.md#function-check_fixed_size_mpi_type)`<br> |
|  [**int**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**layers**](#variable-layers)   = `1`<br> |
|  [**detail::LockFreeRMALevel**](classdynampi_1_1detail_1_1LockFreeRMALevel.md)&lt; [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; \* | [**parent**](#variable-parent)  <br> |
|  std::deque&lt; PendingRelay &gt; | [**pending\_relays**](#variable-pending_relays)  <br> |
|  [**int64\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**pending\_task\_count**](#variable-pending_task_count)   = `0`<br> |
|  std::vector&lt; [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; | [**relay\_buffer**](#variable-relay_buffer)  <br> |
|  [**bool**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**relay\_error\_pending**](#variable-relay_error_pending)   = `[**false**](namespacedynampi.md#function-check_fixed_size_mpi_type)`<br> |
|  [**int64\_t**](namespacedynampi.md#function-check_fixed_size_mpi_type) | [**total\_claimed**](#variable-total_claimed)   = `0`<br> |
















## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**BridgeHop**](#function-bridgehop) ([**detail::LockFreeRMALevel**](classdynampi_1_1detail_1_1LockFreeRMALevel.md)&lt; [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; \* p, [**detail::LockFreeRMALevel**](classdynampi_1_1detail_1_1LockFreeRMALevel.md)&lt; [**TaskT**](namespacedynampi.md#function-check_fixed_size_mpi_type), [**ResultT**](namespacedynampi.md#function-check_fixed_size_mpi_type) &gt; \* c) <br> |




























## Public Attributes Documentation




### variable child 

```C++
detail::LockFreeRMALevel<TaskT, ResultT>* dynampi::HierarchicalLockFreeRMAWorkDistributor< TaskT, ResultT, Options >::BridgeHop::child;
```




<hr>



### variable finish\_marked 

```C++
bool dynampi::HierarchicalLockFreeRMAWorkDistributor< TaskT, ResultT, Options >::BridgeHop::finish_marked;
```




<hr>



### variable layers 

```C++
int dynampi::HierarchicalLockFreeRMAWorkDistributor< TaskT, ResultT, Options >::BridgeHop::layers;
```




<hr>



### variable parent 

```C++
detail::LockFreeRMALevel<TaskT, ResultT>* dynampi::HierarchicalLockFreeRMAWorkDistributor< TaskT, ResultT, Options >::BridgeHop::parent;
```




<hr>



### variable pending\_relays 

```C++
std::deque<PendingRelay> dynampi::HierarchicalLockFreeRMAWorkDistributor< TaskT, ResultT, Options >::BridgeHop::pending_relays;
```




<hr>



### variable pending\_task\_count 

```C++
int64_t dynampi::HierarchicalLockFreeRMAWorkDistributor< TaskT, ResultT, Options >::BridgeHop::pending_task_count;
```




<hr>



### variable relay\_buffer 

```C++
std::vector<ResultT> dynampi::HierarchicalLockFreeRMAWorkDistributor< TaskT, ResultT, Options >::BridgeHop::relay_buffer;
```




<hr>



### variable relay\_error\_pending 

```C++
bool dynampi::HierarchicalLockFreeRMAWorkDistributor< TaskT, ResultT, Options >::BridgeHop::relay_error_pending;
```




<hr>



### variable total\_claimed 

```C++
int64_t dynampi::HierarchicalLockFreeRMAWorkDistributor< TaskT, ResultT, Options >::BridgeHop::total_claimed;
```




<hr>
## Public Functions Documentation




### function BridgeHop 

```C++
inline BridgeHop::BridgeHop (
    detail::LockFreeRMALevel < TaskT , ResultT > * p,
    detail::LockFreeRMALevel < TaskT , ResultT > * c
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/hierarchical_lockfree_rma_distributor.hpp`

