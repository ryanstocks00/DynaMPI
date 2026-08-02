# Aurora benchmark environment (reproducibility reference)

Captured 2026-08-02 for the strong-scaling / shutdown-time result sweep in
this branch. Two sources: the login node used for job submission/building,
and a live compute-node allocation (job 8728418) used to capture hardware
and driver details that only exist on compute nodes. Raw captured output
lives in `raw_job_logs/dynampi_capture_env.o8728418`.

## System

- Machine: ALCF Aurora (Intel/HPE Sapphire Rapids + Ponte Vecchio)
- OS: SUSE Linux Enterprise Server 15 SP4 (`15.4`)
- Kernel: `5.14.21-150400.24.225-default` (SMP PREEMPT_DYNAMIC, built 2026-06-17)
- Compute node hostname sampled: `x4016c5s1b0n0`

## CPU (per compute node)

- 2x Intel(R) Xeon(R) CPU Max 9470C ("Sapphire Rapids HBM" -- `cputype=SPRHBM`
  per `pbsnodes`), 52 cores/socket, 2 threads/core -> 208 logical CPUs
- 4 NUMA nodes: 2 CPU-bearing (0-51,104-155 and 52-103,156-207, ~515 GB each)
  + 2 memory-only HBM nodes (~65 GB each, no CPUs attached)
- Total system memory: ~1.1 TiB (`MemTotal: 1189456432 kB`)
- No swap configured

Full `lscpu` output (cache sizes, flags, mitigation status) is in
`raw_job_logs/dynampi_capture_env.o8728418`.

## GPU (per compute node)

- 6x Intel(R) Data Center GPU Max 1550 ("Ponte Vecchio", `gputype=PVC` per
  `pbsnodes`)
- Level-Zero driver: `12.60.7 [1.6.33578+42]` (via `sycl-ls`)
- OpenCL: 2 platforms available (Intel(R) OpenCL, OpenCL 3.0 LINUX)
- `xpu-smi` was not on PATH in this environment/module set -- Level-Zero
  device enumeration via `sycl-ls`/`clinfo` was used instead (full output in
  the raw log)
- **Note**: all benchmark runs in this sweep are CPU/MPI-only (no GPU
  kernels) -- "rpn=7" (7 ranks/node) was chosen to match 1 rank per GPU for
  a GPU-density-matched network/topology comparison against "rpn=102"
  (1 rank per CPU core), not because the benchmarked code touches the GPUs.

## Interconnect / MPI

- Fabric: Slingshot via libfabric `1.22.0` (`fi_info --version`), CXI provider
- `FI_CXI_RX_MATCH_MODE=software` was set for every job in this sweep (see
  Job configuration below)
- MPI: MPICH `5.0.0b1` (`mpichversion`), ABI `17:0:5`, built from
  `mpich/opt/5.0.0.aurora_test.3c70a61`
- Launcher: `mpiexec` (Cray PALS) version `1.8.0`, revision `a3475e5c9cce`,
  built Aug 25 2025, via modules `cray-pals/1.8.0` + `cray-libpals/1.8.0`

## Filesystem

- Working tree / build / job I/O: `/home/ryans/DynaMPI` (NFS-backed home)
- `--filesystems flare` requested on every job: `/lus/flare`, a 91 PB Lustre
  filesystem (`172.22.12.130@o2ib21:172.22.12.131@o2ib21:/grand`)

## Toolchain / build

- Compiler: Intel oneAPI DPC++/C++ `2025.3.2` (`icx`/`icpx`,
  `2025.3.2.20260112`), via module `oneapi/release/2025.3.1`
- Also loaded: `gcc/13.4.0` (spack-built; provides libstdc++/binutils, not
  the compiler actually used for DynaMPI's own sources -- CMake selected
  `icpx`/`icx`)
- CMake: `3.31.11` (module `cmake/3.31.11`; not on PATH by default, must be
  `module load`ed)
- Build type: `Release` (`-O3 -DNDEBUG`, no debug/assert overhead)
- MPI found by CMake: MPICH `v4.1()` API level, headers/libs from
  `mpich-5.0.0.aurora_test.3c70a61-hlkigtk`

## Module environment (identical on login node and compute nodes)

```
1) gcc/13.4.0                      4) libfabric/1.22.0
2) oneapi/release/2025.3.1         5) cray-pals/1.8.0
3) mpich/opt/5.0.0.aurora_test.3c70a61   6) cray-libpals/1.8.0
```

## Job configuration common to every run in this sweep

- `#PBS -l filesystems=flare`
- `export FI_CXI_RX_MATCH_MODE=software` (set before every `mpiexec` call --
  switches the CXI provider's tag-matching from hardware-offloaded to
  software-emulated matching; used throughout this whole effort, not
  specific to any one benchmark)
- Queues used depending on node count: `debug` (<=2 nodes), `debug-scaling`
  (small/quick jobs, tightly rate-limited to ~1 job at a time for this
  account), `prod`/`small`/`large` (>=256 nodes, routed automatically by
  PBS based on node count)

## Git provenance

- Results in this branch were produced from commits on
  `review/async-put-lockfree-distributors`, HEAD at branch time:
  `e0f9321` ("Add final missing 2048-node combo (async_put_lockfree crash
  retry)")
- **Two distinct binary builds contributed data, both functionally
  equivalent for the benchmarked code paths:**
  - Build A (commit `e842a26`, built 2026-07-31 18:08): used for the large
    majority of the sweep -- the MPI_Abort-based termination fix and the
    shutdown_time finalize()-timing fix, which most of this dataset depends
    on for correctness, were introduced here.
  - Build B (commit `64e71b1`, built 2026-08-02 03:10): used only for the
    final few jobs (2048-node strong-scaling completion, the crash retry,
    and this environment-capture job). Picked up 9 unrelated upstream
    commits (Frontier launch-script parity, macOS/coverage CI fixes, a
    `-Wmaybe-uninitialized` fix) plus two small changes to the benchmark
    drivers themselves: a cosmetic reformat in
    `strong_scaling_distribution_rate.cpp`, and a correctness fix in
    `shutdown_time.cpp` (replaced a `rank==0` check with a
    `distributor.is_root_manager()`-derived flag for which iterations get
    timed -- doesn't change results for any config in this sweep, all of
    which use `manager_rank=0`).
- Full commit history: see `raw_job_logs/git_provenance.txt` (or
  `git log` on this branch).

## Node-hour accounting

- This sweep ran under ALCF project allocation `DynaMPI` (Aurora resource).
- See PBS accounting (`sbank-list-allocations`) for authoritative charged
  totals; not reproduced here since it changes over time and isn't specific
  to any one run.

## What's in `raw_job_logs/`

Every PBS job script (`*.pbs`) and its captured stdout/stderr (`*.o<jobid>`)
from this results-collection effort, plus the shell orchestrator scripts
(`run_*.sh`) used to submit grouped small-node sweeps while respecting the
`debug-scaling` queue's tight per-account concurrency limit. This includes
both the runs that fed the final CSVs under `benchmark/results/` (on the
`review/async-put-lockfree-distributors` branch) and earlier
investigation/debugging runs (crash repros, hang diagnosis, correctness
checks, fanout-structure exploration) from the same effort -- kept for full
provenance rather than filtered down to only the "successful" runs.
