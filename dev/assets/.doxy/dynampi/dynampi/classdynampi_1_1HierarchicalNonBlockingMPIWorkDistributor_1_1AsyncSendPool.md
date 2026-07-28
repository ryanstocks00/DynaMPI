

# Class dynampi::HierarchicalNonBlockingMPIWorkDistributor::AsyncSendPool

**template &lt;typename T&gt;**



[**ClassList**](annotated.md) **>** [**AsyncSendPool**](classdynampi_1_1HierarchicalNonBlockingMPIWorkDistributor_1_1AsyncSendPool.md)










































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**post**](#function-post) ([**MPICommunicator**](classdynampi_1_1MPICommunicator.md) & comm, T value, int dest, int tag) <br> |
|  void | [**wait\_all**](#function-wait_all) () <br> |




























## Public Functions Documentation




### function post 

```C++
inline void AsyncSendPool::post (
    MPICommunicator & comm,
    T value,
    int dest,
    int tag
) 
```




<hr>



### function wait\_all 

```C++
inline void AsyncSendPool::wait_all () 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/impl/hierarchical_nonblocking_distributor.hpp`

