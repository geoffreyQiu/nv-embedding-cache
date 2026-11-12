# Large-table HostLayer CPU comparison

This suite compares NVIDIA Vera 88-core x2, AMD EPYC 9965 192-core x2 and
AMD EPYC 9755 128-core x2. It measures native NVE HostLayer inference lookup.
PyTorch is part of the container environment, not the measured execution path.

## Workload and timing

| Setting | Value |
|---|---|
| Table | One shared 16 GiB table: 33,554,432 rows x 128 FP32 values |
| Request | 32 x 512 = 16,384 keys, 8 MiB logical output per call |
| Key distributions | Power law alpha=1.15 and uniform alpha=0, reported separately |
| Warmup | At least 2 seconds and 100 calls per caller before each measured phase |
| Throughput phase | 10 seconds, all callers share a start and stop deadline |
| Latency phase | 10 seconds, all callers share a start and stop deadline |
| Repetitions | Five, shuffled case/distribution order each round |
| Key storage | 512 pre-generated batches per caller, 64 MiB, replayed cyclically |
| Pool task threshold | 1,024 keys, at most 16 gather tasks per call |
| GPU / training / pooling | None |

16 GiB exceeds 8 GiB and is checked against the aggregate L3 capacity reported
by sysfs when that information is available. A larger allocation alone does not
prevent the hot rows of a skewed distribution from fitting in cache. The uniform
control distributes requests over the table, reducing that concentration; it
does not guarantee a cache miss for every key. Neither test measures physical
DRAM bandwidth. No cache flush is performed between calls.

Key generation, table initialization, correctness checks, CSV writes and THP
observations are outside the measured intervals. A fresh key ring is generated
and warmed before the latency phase. Rings replay during each phase: these are
repeated finite traces, not unlimited fresh-key streams. Duration does not cause
the key buffers to grow. Input-key memory and output stores still contribute to
the actual workload. The 16 GiB setting is the table payload, not process RSS.

The fixed-step mode remains available for earlier experiments. Timed mode adds
`--throughput_seconds`, `--latency_seconds`, `--warmup_seconds` and
`--key_buffer_steps`. Actual completed calls, time and sample counts are recorded.
Each phase drains calls started before its deadline, so elapsed time can exceed
the requested duration. No caller stops early after completing a fixed quota.
Latency samples are reserved before timing; exceeding the sample limit fails the
run rather than reallocating or silently dropping tail samples.

## CPU and NUMA placement

Let P be physical cores per socket: 88 for Vera, 128 or 192 for Turin.

| Scenario | One socket: callers / NVE workers | Whole server, two sockets: callers / NVE workers |
|---|---|---|
| Inline, all physical cores | P / 0 | 2P / 0 |
| Inline, 64 callers per socket | 64 / 0 | 128 / 0 |
| Pool, 16 callers per socket | 16 / P | 32 / 2P |
| Pool, 32 callers per socket | 32 / P | 64 / 2P |
| Pool, 64 callers per socket | 64 / P | 128 / 2P |

SMT is unused in every scenario. The runner selects one hardware thread per
physical core. Each caller is pinned to a distinct selected CPU. Each pool
worker is pinned to a distinct physical-core CPU, with equal worker counts per
socket. Callers are additional software threads and share cores with pool
workers. CPU choices for fixed caller counts are spread through each socket's
sorted physical-core CPU list; this does not claim optimal cache placement.

There is **one process, one shared table and, when enabled, one shared
SimpleThreadPool** in both scopes. The two-socket experiment uses a global pool
spanning both sockets, not separate socket-local pools. The harness pins the
newly created NVE worker TIDs before issuing any lookup; unexpected thread counts
or affinity failures abort the run. NVE library code is unchanged.

One socket binds memory to its CPU node. Two sockets interleave allocations
across their local memory nodes, as in the earlier shared-table experiment.
If a socket contains multiple memory-bearing NUMA nodes, its allocations are
interleaved across those nodes. This policy applies to **all allocations**, not
only the table: key and output buffers are also affected. The table is not
replicated and remains 16 GiB in the two-socket case.

## Interpreting latency and throughput

- MKPS is aggregate millions of processed key requests per second. Repeated
  keys count each time. Logical payload GB/s equals MKPS x 0.512 for this workload.
- Mean and p99 are for one entire 16,384-key synchronous lookup call. Pool queue
  waits and OS scheduling during that call are included. External service
  queues, request generation and network handling are not included.
- At a fixed scope, compare the 64-caller inline scenario or each pool scenario
  across servers at the same caller count. The full-core scenario has different
  concurrency on each product; label its latency as full-core configuration
  latency rather than equal-load latency.
- Pool-versus-inline at 64 callers per socket matches caller concurrency, but
  the pool uses all physical cores while inline uses 64 cores. These do not
  represent identical hardware budgets.
- The two-socket case doubles total callers while keeping callers per socket
  fixed. Compare scaling with that load change stated.
- This is a closed-loop benchmark: each caller waits for completion before
  issuing another request. Equal concurrency does not impose equal incoming
  QPS. Fixed-arrival-rate service-latency testing would require another harness.
- Per-run latency percentiles combine all caller samples. Summary percentiles
  average the per-run percentiles; they are not percentiles of all five runs
  pooled together. Samples are request-weighted, not equally weighted by caller.

## Fresh allocation, container and build

Choose one allocation command on the login host:

```bash
# Vera
salloc -p arthurq -N 1 --exclusive --mem=0 -t 04:00:00
# Turin 192-core
salloc -p turin192q -N 1 --exclusive --mem=0 -t 04:00:00
# Turin 128-core
salloc -p turin128q -N 1 --exclusive --mem=0 -t 04:00:00
```

Then enter one interactive container:

```bash
srun --ntasks=1 --cpu-bind=none \
  --container-image=nvcr.io/nvidia/pytorch:26.03-py3 --pty bash -l
```

Inside the container:

```bash
cd /tmp
git clone --branch codex/hostlayer-cpu-inference-benchmark \
  https://github.com/geoffreyQiu/nv-embedding-cache.git
cd nv-embedding-cache
export CUDA_VISIBLE_DEVICES=-1
BUILD_JOBS=8 bash benchmarks/host_layer/build.sh

python3 tests/benchmarks/test_host_layer_bench.py build-host-layer/bin/host_layer_bench
python3 tests/benchmarks/test_host_layer_multicore.py
python3 tests/benchmarks/test_host_layer_scaled.py
python3 benchmarks/host_layer/scaled.py --smoke
```

The small smoke run checks the real binary, timed phases, NUMA policy and exact
CPU pinning in the current container. It is not performance data. The full suite
rejects incomplete CPU access, unexpected topology, visible resource limits,
stale binary revisions and modified tracked source. It does not require old
results, copied CPU masks, sudo or a Docker daemon inside the container.

## Run and collect

```bash
# Inspect the ten CPU/NUMA configurations without allocating the large table.
python3 benchmarks/host_layer/scaled.py --plan-only

# 10 configurations x 2 distributions x 5 repeats = 100 measured runs.
python3 benchmarks/host_layer/scaled.py --seconds 10 --repeats 5
```

The measured phases total at least 2,000 seconds (33 minutes 20 seconds).
Warmup adds at least 400 seconds. Building, table initialization, key generation
and other checks take additional time. All cases run sequentially; the runner
does not launch multiple competing benchmark processes.

Every invocation creates a new directory in `build-host-layer/results/`, with:

- `environment.json`: product, software, kernel, L3 capacity, governor, page
  size, NUMA and THP configuration, and visible cgroup limits.
- `plan.json`, `topology.csv`: exact per-caller/per-worker CPU IDs, NUMA policy,
  workload, repeat count, status and completed-run count.
- Native per-case CSVs and per-run logs: exact commands and full measurements.
- `runs.csv`, `summary.csv`: validated individual runs and grouped means.
- `results-for-chat.txt`: compact environment, results and per-run throughput.

The runner prints the final compact report automatically. To display it again:

```bash
python3 benchmarks/host_layer/scaled.py --report latest
```

The report command can also show completed runs from an interrupted suite;
check its status and run counts before treating it as a complete comparison.
Pass a printed results directory instead of `latest` to select a specific run.
Smoke and plan-only invocations do not replace the latest-full-run pointer.
Files under `/tmp` are ephemeral. When export is unavailable, paste the compact
report into the conversation before releasing the allocation.

Host THP settings are recorded and left unchanged. Native timed runs read
process-wide `AnonHugePages` after each measured phase, while buffers are still
alive and all callers are outside timing. These are two snapshots, not continuous
monitoring or a table-specific attribution. No THP=always experiment is required.
