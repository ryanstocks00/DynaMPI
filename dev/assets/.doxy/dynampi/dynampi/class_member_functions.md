
# Class Member Functions



## a

* **average\_receive\_size** ([**dynampi::CommStatistics**](structdynampi_1_1CommStatistics.md))
* **average\_send\_size** ([**dynampi::CommStatistics**](structdynampi_1_1CommStatistics.md))
* **allocate\_task\_to\_child** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md))
* **atomic\_read** ([**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md))
* **atomic\_set** ([**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md))
* **available** ([**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md))
* **active\_worker\_count** ([**dynampi::NaiveMPIWorkDistributor**](classdynampi_1_1NaiveMPIWorkDistributor.md))


## b

* **broadcast** ([**dynampi::MPICommunicator**](classdynampi_1_1MPICommunicator.md))
* **broadcast\_done** ([**dynampi::NaiveMPIWorkDistributor**](classdynampi_1_1NaiveMPIWorkDistributor.md))


## c

* **create\_statistics** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md), [**dynampi::NaiveMPIWorkDistributor**](classdynampi_1_1NaiveMPIWorkDistributor.md))
* **compare\_and\_swap** ([**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md))
* **contains\_rank\_in\_group** ([**dynampi::MPIGroup**](classdynampi_1_1MPIGroup.md))
* **count** ([**dynampi::MPI\_Type&lt; std::nullptr\_t &gt;**](structdynampi_1_1MPI__Type_3_01std_1_1nullptr__t_01_4.md), [**dynampi::MPI\_Type&lt; std::string &gt;**](structdynampi_1_1MPI__Type_3_01std_1_1string_01_4.md), [**dynampi::MPI\_Type&lt; std::vector&lt; T &gt;, std::enable\_if\_t&lt; has\_dynampi\_mpi\_type&lt; T &gt;::value &gt; &gt;**](structdynampi_1_1MPI__Type_3_01std_1_1vector_3_01T_01_4_00_01std_1_1enable__if__t_3_01has__dynam0c05b0754f90b71498257126104ee051.md))
* **collect\_available\_results** ([**dynampi::NaiveMPIWorkDistributor**](classdynampi_1_1NaiveMPIWorkDistributor.md))


## d

* **determine\_layer\_from\_world\_rank** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md))
* **drain\_results** ([**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md))


## e

* **exchange\_gathered\_results** ([**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md))
* **ensure\_result\_capacity** ([**dynampi::NaiveMPIWorkDistributor**](classdynampi_1_1NaiveMPIWorkDistributor.md))
* **elapsed** ([**dynampi::Timer**](classdynampi_1_1Timer.md))


## f

* **finalize** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md), [**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md), [**dynampi::NaiveMPIWorkDistributor**](classdynampi_1_1NaiveMPIWorkDistributor.md))
* **finish\_remaining\_tasks** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md), [**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md), [**dynampi::NaiveMPIWorkDistributor**](classdynampi_1_1NaiveMPIWorkDistributor.md))
* **flush** ([**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md))
* **fetch\_add** ([**dynampi::MinimalLockFreeMPIWorkDistributor**](classdynampi_1_1MinimalLockFreeMPIWorkDistributor.md))


## g

* **get\_next\_task\_to\_send** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md))
* **get\_parent\_target** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md))
* **get\_statistics** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md), [**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md), [**dynampi::MPICommunicator**](classdynampi_1_1MPICommunicator.md), [**dynampi::NaiveMPIWorkDistributor**](classdynampi_1_1NaiveMPIWorkDistributor.md))
* **get\_bytes** ([**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md))
* **gather** ([**dynampi::MPICommunicator**](classdynampi_1_1MPICommunicator.md))
* **get** ([**dynampi::MPICommunicator**](classdynampi_1_1MPICommunicator.md), [**dynampi::MPIGroup**](classdynampi_1_1MPIGroup.md))
* **get\_group** ([**dynampi::MPICommunicator**](classdynampi_1_1MPICommunicator.md))
* **gather\_sorted** ([**dynampi::MinimalLockFreeMPIWorkDistributor**](classdynampi_1_1MinimalLockFreeMPIWorkDistributor.md))


## h

* **HierarchicalMPIWorkDistributor** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md))
* **handle\_result\_message** ([**dynampi::NaiveMPIWorkDistributor**](classdynampi_1_1NaiveMPIWorkDistributor.md))


## i

* **idx\_for\_worker** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md))
* **insert\_task** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md), [**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md), [**dynampi::NaiveMPIWorkDistributor**](classdynampi_1_1NaiveMPIWorkDistributor.md))
* **insert\_tasks** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md), [**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md), [**dynampi::NaiveMPIWorkDistributor**](classdynampi_1_1NaiveMPIWorkDistributor.md))
* **is\_leaf\_worker** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md))
* **is\_root\_manager** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md), [**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md), [**dynampi::MinimalLockFreeMPIWorkDistributor**](classdynampi_1_1MinimalLockFreeMPIWorkDistributor.md), [**dynampi::NaiveMPIWorkDistributor**](classdynampi_1_1NaiveMPIWorkDistributor.md))
* **initialize\_window** ([**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md))
* **iprobe** ([**dynampi::MPICommunicator**](classdynampi_1_1MPICommunicator.md))


## l

* **LockFreeMPIWorkDistributor** ([**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md))


## m

* **max\_workers\_per\_coordinator** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md))
* **make\_statistics** ([**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md))
* **maybe\_participate\_in\_gather** ([**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md))
* **MPICommunicator** ([**dynampi::MPICommunicator**](classdynampi_1_1MPICommunicator.md))
* **MPIGroup** ([**dynampi::MPIGroup**](classdynampi_1_1MPIGroup.md))
* **MinimalLockFreeMPIWorkDistributor** ([**dynampi::MinimalLockFreeMPIWorkDistributor**](classdynampi_1_1MinimalLockFreeMPIWorkDistributor.md))


## n

* **num\_direct\_children** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md))
* **num\_workers** ([**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md), [**dynampi::NaiveMPIWorkDistributor**](classdynampi_1_1NaiveMPIWorkDistributor.md))
* **NaiveMPIWorkDistributor** ([**dynampi::NaiveMPIWorkDistributor**](classdynampi_1_1NaiveMPIWorkDistributor.md))


## o

* **operator MPI\_Comm** ([**dynampi::MPICommunicator**](classdynampi_1_1MPICommunicator.md))
* **operator=** ([**dynampi::MPICommunicator**](classdynampi_1_1MPICommunicator.md), [**dynampi::MPIGroup**](classdynampi_1_1MPIGroup.md))
* **operator MPI\_Group** ([**dynampi::MPIGroup**](classdynampi_1_1MPIGroup.md))


## p

* **publish\_task** ([**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md))
* **put\_bytes** ([**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md))
* **probe** ([**dynampi::MPICommunicator**](classdynampi_1_1MPICommunicator.md))
* **ptr** ([**dynampi::MPI\_Type&lt; std::nullptr\_t &gt;**](structdynampi_1_1MPI__Type_3_01std_1_1nullptr__t_01_4.md), [**dynampi::MPI\_Type&lt; std::string &gt;**](structdynampi_1_1MPI__Type_3_01std_1_1string_01_4.md), [**dynampi::MPI\_Type&lt; std::vector&lt; T &gt;, std::enable\_if\_t&lt; has\_dynampi\_mpi\_type&lt; T &gt;::value &gt; &gt;**](structdynampi_1_1MPI__Type_3_01std_1_1vector_3_01T_01_4_00_01std_1_1enable__if__t_3_01has__dynam0c05b0754f90b71498257126104ee051.md))
* **pop\_next\_task** ([**dynampi::NaiveMPIWorkDistributor**](classdynampi_1_1NaiveMPIWorkDistributor.md))
* **process\_incoming\_message** ([**dynampi::NaiveMPIWorkDistributor**](classdynampi_1_1NaiveMPIWorkDistributor.md))


## r

* **reset** ([**dynampi::CommStatistics**](structdynampi_1_1CommStatistics.md), [**dynampi::Timer**](classdynampi_1_1Timer.md))
* **receive\_done\_from** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md))
* **receive\_execute\_return\_task\_from** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md))
* **receive\_from\_anyone** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md))
* **receive\_request\_batch\_from** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md))
* **receive\_request\_from** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md))
* **receive\_result\_batch\_from** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md))
* **receive\_result\_from** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md))
* **receive\_task\_batch\_from** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md))
* **remaining\_tasks\_count** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md), [**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md), [**dynampi::NaiveMPIWorkDistributor**](classdynampi_1_1NaiveMPIWorkDistributor.md))
* **return\_results\_and\_request\_next\_batch\_from\_manager** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md))
* **run\_tasks** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md), [**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md), [**dynampi::NaiveMPIWorkDistributor**](classdynampi_1_1NaiveMPIWorkDistributor.md))
* **run\_worker** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md), [**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md), [**dynampi::NaiveMPIWorkDistributor**](classdynampi_1_1NaiveMPIWorkDistributor.md))
* **read\_task** ([**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md))
* **request\_gather** ([**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md))
* **run\_one\_task\_locally** ([**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md))
* **rank** ([**dynampi::MPICommunicator**](classdynampi_1_1MPICommunicator.md), [**dynampi::MPIGroup**](classdynampi_1_1MPIGroup.md))
* **recv** ([**dynampi::MPICommunicator**](classdynampi_1_1MPICommunicator.md))
* **recv\_any** ([**dynampi::MPICommunicator**](classdynampi_1_1MPICommunicator.md))
* **recv\_empty** ([**dynampi::MPICommunicator**](classdynampi_1_1MPICommunicator.md))
* **recv\_empty\_message** ([**dynampi::MPICommunicator**](classdynampi_1_1MPICommunicator.md))
* **resize** ([**dynampi::MPI\_Type&lt; std::nullptr\_t &gt;**](structdynampi_1_1MPI__Type_3_01std_1_1nullptr__t_01_4.md), [**dynampi::MPI\_Type&lt; std::string &gt;**](structdynampi_1_1MPI__Type_3_01std_1_1string_01_4.md), [**dynampi::MPI\_Type&lt; std::vector&lt; T &gt;, std::enable\_if\_t&lt; has\_dynampi\_mpi\_type&lt; T &gt;::value &gt; &gt;**](structdynampi_1_1MPI__Type_3_01std_1_1vector_3_01T_01_4_00_01std_1_1enable__if__t_3_01has__dynam0c05b0754f90b71498257126104ee051.md))
* **run** ([**dynampi::MinimalLockFreeMPIWorkDistributor**](classdynampi_1_1MinimalLockFreeMPIWorkDistributor.md))
* **rank\_to\_worker\_idx** ([**dynampi::NaiveMPIWorkDistributor**](classdynampi_1_1NaiveMPIWorkDistributor.md))
* **run\_task\_locally** ([**dynampi::NaiveMPIWorkDistributor**](classdynampi_1_1NaiveMPIWorkDistributor.md))


## s

* **send\_done\_to\_children\_when\_free** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md))
* **send\_to\_parent** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md))
* **send\_to\_worker** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md))
* **store\_result** ([**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md))
* **send** ([**dynampi::MPICommunicator**](classdynampi_1_1MPICommunicator.md))
* **send\_empty** ([**dynampi::MPICommunicator**](classdynampi_1_1MPICommunicator.md))
* **size** ([**dynampi::MPICommunicator**](classdynampi_1_1MPICommunicator.md), [**dynampi::MPIGroup**](classdynampi_1_1MPIGroup.md))
* **split** ([**dynampi::MPICommunicator**](classdynampi_1_1MPICommunicator.md))
* **split\_by\_node** ([**dynampi::MPICommunicator**](classdynampi_1_1MPICommunicator.md))
* **set\_counter** ([**dynampi::MinimalLockFreeMPIWorkDistributor**](classdynampi_1_1MinimalLockFreeMPIWorkDistributor.md))
* **send\_next\_task\_to\_worker** ([**dynampi::NaiveMPIWorkDistributor**](classdynampi_1_1NaiveMPIWorkDistributor.md))
* **start** ([**dynampi::Timer**](classdynampi_1_1Timer.md))
* **stop** ([**dynampi::Timer**](classdynampi_1_1Timer.md))


## t

* **total\_num\_children** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md))
* **task\_slot** ([**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md))
* **try\_gather\_results** ([**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md))
* **translate\_rank** ([**dynampi::MPIGroup**](classdynampi_1_1MPIGroup.md))
* **translate\_ranks** ([**dynampi::MPIGroup**](classdynampi_1_1MPIGroup.md))
* **Timer** ([**dynampi::Timer**](classdynampi_1_1Timer.md))


## u

* **update\_contiguous\_results\_count** ([**dynampi::NaiveMPIWorkDistributor**](classdynampi_1_1NaiveMPIWorkDistributor.md))


## w

* **worker\_for\_idx** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md))
* **worker\_idx\_to\_rank** ([**dynampi::NaiveMPIWorkDistributor**](classdynampi_1_1NaiveMPIWorkDistributor.md))


## ~

* **~HierarchicalMPIWorkDistributor** ([**dynampi::HierarchicalMPIWorkDistributor**](classdynampi_1_1HierarchicalMPIWorkDistributor.md))
* **~LockFreeMPIWorkDistributor** ([**dynampi::LockFreeMPIWorkDistributor**](classdynampi_1_1LockFreeMPIWorkDistributor.md))
* **~MPICommunicator** ([**dynampi::MPICommunicator**](classdynampi_1_1MPICommunicator.md))
* **~MPIGroup** ([**dynampi::MPIGroup**](classdynampi_1_1MPIGroup.md))
* **~MinimalLockFreeMPIWorkDistributor** ([**dynampi::MinimalLockFreeMPIWorkDistributor**](classdynampi_1_1MinimalLockFreeMPIWorkDistributor.md))
* **~NaiveMPIWorkDistributor** ([**dynampi::NaiveMPIWorkDistributor**](classdynampi_1_1NaiveMPIWorkDistributor.md))




