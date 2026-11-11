# Socket and whole-server HostLayer benchmark

Run from the repository root inside the existing development container. No
Docker daemon or file transfer into the container is needed. Use the same
container release and source revision on all servers.

```bash
git fetch origin codex/hostlayer-cpu-inference-benchmark
git switch --detach FETCH_HEAD
export CUDA_VISIBLE_DEVICES=-1
BUILD_JOBS=8 bash benchmarks/host_layer/build.sh
python3 tests/benchmarks/test_host_layer_bench.py build-host-layer/bin/host_layer_bench
```

These fetch commands assume `origin` is the user's benchmark fork. Otherwise,
fetch from `https://github.com/geoffreyQiu/nv-embedding-cache.git` explicitly.
The runner checks that the binary's embedded source revision matches the checkout
before starting the large experiments. Rebuild after checking out a new commit.

## Run

```bash
# Preview CPU counts, memory policies and buffer sizes.
python3 benchmarks/host_layer/multicore.py --mode inline --smt --plan-only

# Five repetitions of the single-core, socket and server cases, including SMT.
python3 benchmarks/host_layer/multicore.py --mode inline --smt --repeats 5

# Optional shared-worker-pool comparison using physical cores.
python3 benchmarks/host_layer/multicore.py --mode pool --repeats 5
```

The fixed workload is one shared table of 1,000,000 rows x 128 FP32 elements,
16,384 keys per lookup, alpha 1.15, and 100 warmup / 1,000 throughput / 200
latency calls per caller. Each repetition shuffles the execution order.

| Case | Callers | CPU and memory placement |
|---|---:|---|
| Single-core (inline only) | 1 | One logical CPU; its local memory node |
| Socket cores | Physical cores in the lowest-numbered socket | One logical CPU per core; socket-local memory nodes |
| Server cores | All physical cores | One logical CPU per core across all sockets; their local memory nodes |
| Socket SMT (`--smt`) | All logical CPUs in that socket | All hardware threads; socket-local memory nodes |
| Server SMT (`--smt`) | All logical CPUs | All hardware threads and their local memory nodes |

A single memory node uses `membind`; multiple nodes use `interleave`. The latter
policy applies to the table, keys, outputs and other allocations. There is one
shared table per process, not a replica per socket. CPU masks restrict the whole
process; individual threads can migrate within the mask. Memoryless CPU nodes
require explicit topology review rather than guessing their nearest memory.

Inline mode executes on the callers with the NVE pool disabled. Pool mode keeps
the same caller count and adds one shared pool with that many workers; workers
are additional software threads. This is an initial configuration, not a claim
of optimal pool sizing. With the default task threshold of 1,024, each
16,384-key request creates at most 16 gather tasks. One caller with hundreds of
workers therefore does not exercise a whole socket.

An exclusive whole-node allocation should expose all required CPUs and enough
memory. For example, request `--mem=0` at allocation time and use
`srun --nodes=1 --ntasks=1 --cpu-bind=none` to enter the container. The largest
768-caller case requires about 103 GiB for the main buffers, plus overhead.
The runner rejects incomplete CPU access and checks visible cgroup v2 limits.
Host or ancestor-cgroup limits may impose additional restrictions.

## Results

Each invocation creates a unique directory under `build-host-layer/results/`
containing `environment.txt`, `topology.csv`, `plan.json`, per-run logs, per-case
CSVs and `summary.csv`. Logs contain the exact benchmark commands. Summary
throughput is aggregate effective NVE key throughput; effective payload GB/s is
not a measurement of DRAM traffic. Latency is under concurrent load. Per-run
percentiles combine caller samples; summary percentiles average the per-run
percentiles.

When file export from the container is unavailable, display and copy the small
summary CSV and selected environment lines through the terminal. A tar archive
left only in `/tmp` still disappears when that storage is released. Preserve
individual run CSVs externally when possible to retain variability information.

## Later THP round

The runner prints a reminder to compare THP `madvise` with `always`. It does not
change host settings. Have a host administrator change `enabled` to `always`
and ensure the PMD-size policy inherits it. Keep defrag and smaller-page policies
unchanged for the initial comparison, record the settings, and restart the
benchmark into a new results directory. Measure actual THP usage separately;
`always` does not guarantee huge pages. This matters particularly for Vera's
512 MiB PMD size and the benchmark's 488.28 MiB table.
