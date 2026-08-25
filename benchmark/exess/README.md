<!--
SPDX-FileCopyrightText: 2026 Ryan Stocks
SPDX-License-Identifier: Apache-2.0
-->

# EXESS case study

The three runs behind the EXESS case study: an MBE3 RI-HF energy calculation on
`(H2O)393`, run once with each of three DynaMPI distributors.  Each directory
holds the EXESS input, the Slurm submission script, and the job's stdout.

## Configuration

Every run used the same binary (`exess_rocm713_mpich910`, ROCm 7.1.3 with
cray-mpich 9.1.0) and the same environment setup script.  The three
`input.json` files are byte-identical apart from a single key,
`keywords.frag.work_distributor` — so the comparison is controlled.

| | |
|---|---|
| Nodes | 8192 (Frontier) |
| Ranks per node | 17 (`--ntasks-per-node=17 --cpus-per-task=3`) |
| Worker teams per node | 8, one GCD each |
| GPU workers | 65,536 |
| EXESS world communicator | 131,073 ranks |
| Fragments | 393 monomers + 77,028 dimers + 10,039,316 trimers = 10,116,737 |

The node count is **not** recorded in the submission scripts — they compute
`--ntasks` from `SLURM_JOB_NUM_NODES`, which came from the `sbatch` command
line.  It is recoverable from the logs: with 16 in-world ranks per node plus one
manager, `16N + 1 = 131073` gives `N = 8192`.

Fragmentation cutoffs are set (dimer 1000, trimer 100) but are larger than the
extent of the cluster, so nothing is excluded: the dimer and trimer counts are
exactly `C(393,2)` and `C(393,3)`.

## Results

Timings are the `Result harvesting` line of each log — remaining dispatch, all
worker compute, and result collection.  Task enumeration (~5.5 s in every run)
is excluded, as is job startup.

| Distributor | Job | Harvest | of which `run_tasks` | Result processing | Speedup |
|---|---|--:|--:|--:|--:|
| `DynaMPINaive` | 5336059 | 100.288 s | 95.332 s | 4.957 s | 1.0x |
| `DynaMPIHierarchical` | 5336307 | 19.860 s | 14.580 s | 5.280 s | 5.05x |
| `DynaMPIHierarchicalAsyncPutLockFree` | 5325215 | 7.389 s | 2.202 s | 5.187 s | 13.57x |

All three converge to the same energy (-29877.781 Eh) and report the same total
work (15.13844 PFLOP DP), which is the check that the comparison is like for
like.

At ~40 ms per trimer on one GCD, the ideal wall time across 65,536 workers is
6.13 s, so the 7.389 s run is at 82.9 % efficiency.

## Caveats

- **Each configuration was run once.**  There are no repeats.
- The `hierarchical_rma` run was submitted separately (job 5325215, about an
  hour earlier) and its script differs from the other two in launch mechanics
  only: it runs the binary directly rather than through a `bash -c` wrapper,
  uses a different working directory, and omits `ulimit -c unlimited`.  The
  naive and hierarchical runs were a back-to-back pair.
