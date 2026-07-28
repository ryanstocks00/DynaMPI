

# Struct dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor::BridgeHop



[**ClassList**](annotated.md) **>** [**BridgeHop**](structdynampi_1_1HierarchicalAsyncPutLockFreeMPIWorkDistributor_1_1BridgeHop.md)


























## Public Attributes

| Type | Name |
| ---: | :--- |
|  [**detail::AsyncPutLevel**](classdynampi_1_1detail_1_1AsyncPutLevel.md)&lt; TaskT, ResultT &gt; \* | [**child**](#variable-child)  <br> |
|  bool | [**finish\_marked**](#variable-finish_marked)   = `false`<br> |
|  [**detail::AsyncPutLevel**](classdynampi_1_1detail_1_1AsyncPutLevel.md)&lt; TaskT, ResultT &gt; \* | [**parent**](#variable-parent)  <br> |
|  std::deque&lt; PendingRelay &gt; | [**pending\_relays**](#variable-pending_relays)  <br> |
|  int64\_t | [**pending\_task\_count**](#variable-pending_task_count)   = `0`<br> |
|  std::vector&lt; ResultT &gt; | [**relay\_buffer**](#variable-relay_buffer)  <br> |
















## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**BridgeHop**](#function-bridgehop) ([**detail::AsyncPutLevel**](classdynampi_1_1detail_1_1AsyncPutLevel.md)&lt; TaskT, ResultT &gt; \* p, [**detail::AsyncPutLevel**](classdynampi_1_1detail_1_1AsyncPutLevel.md)&lt; TaskT, ResultT &gt; \* c) <br> |




























## Public Attributes Documentation




### variable child 

```C++
detail::AsyncPutLevel<TaskT, ResultT>* dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor< TaskT, ResultT, Options >::BridgeHop::child;
```




<hr>



### variable finish\_marked 

```C++
bool dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor< TaskT, ResultT, Options >::BridgeHop::finish_marked;
```




<hr>



### variable parent 

```C++
detail::AsyncPutLevel<TaskT, ResultT>* dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor< TaskT, ResultT, Options >::BridgeHop::parent;
```




<hr>



### variable pending\_relays 

```C++
std::deque<PendingRelay> dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor< TaskT, ResultT, Options >::BridgeHop::pending_relays;
```




<hr>



### variable pending\_task\_count 

```C++
int64_t dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor< TaskT, ResultT, Options >::BridgeHop::pending_task_count;
```




<hr>



### variable relay\_buffer 

```C++
std::vector<ResultT> dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor< TaskT, ResultT, Options >::BridgeHop::relay_buffer;
```




<hr>
## Public Functions Documentation




### function BridgeHop 

```C++
inline BridgeHop::BridgeHop (
    detail::AsyncPutLevel < TaskT, ResultT > * p,
    detail::AsyncPutLevel < TaskT, ResultT > * c
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/hierarchical_async_put_lockfree_distributor.hpp`

