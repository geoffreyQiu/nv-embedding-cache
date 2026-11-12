# HostLayer CPU lookup benchmark

CPU inference through `HostEmbeddingLayer<int64_t>` + `LinearHostTable<int64_t>`.
The harness mirrors `single_layer_bench.cpp`: the same power-law generator,
lookup API, fixed-count throughput/latency loops and metric definitions, using
host buffers and synchronous CPU completion. No training or embedding pooling.

## Build and test

Allocate one exclusive node; use `arthurq`, `turin192q` or `turin128q`:

```bash
salloc -p arthurq -N 1 --exclusive --mem=0 -t 04:00:00
srun --ntasks=1 --cpu-bind=none \
  --container-image=nvcr.io/nvidia/pytorch:26.03-py3 --pty bash -l
```

Inside that container:

```bash
cd /tmp
git clone --branch codex/hostlayer-cpu-inference-benchmark \
  https://github.com/geoffreyQiu/nv-embedding-cache.git
cd nv-embedding-cache
export CUDA_VISIBLE_DEVICES=-1
BUILD_JOBS=8 bash benchmarks/host_layer/build.sh
python3 tests/benchmarks/test_host_layer_bench.py build-host-layer/bin/host_layer_bench
python3 tests/benchmarks/test_host_layer_runner.py
python3 benchmarks/host_layer/run.py --smoke
```

Rebuild after changing revisions. The CUDA development image is needed to build
NVE, but the benchmark needs no GPU or Docker daemon inside the container.

## Three scenarios, one socket and two sockets

Defaults: **one shared 16 GiB table**, 33,554,432 rows × 128 FP32 values;
**power law alpha=1.15**; 32 × 512 = **16,384 keys per lookup**. P denotes
physical cores/socket: Vera 88, Turin 128 or Turin 192.

| Case | Total callers | Total NVE workers |
|---|---:|---:|
| `socket-inline-full` | P | 0 |
| `server-inline-full` | 2P | 0 |
| `socket-inline-64` | 64 | 0 |
| `server-inline-64` | 128 | 0 |
| `socket-pool-16` | 16 | P |
| `server-pool-16` | 32 | 2P |

Every caller and worker is pinned to one selected hardware thread per physical
core; SMT siblings are unused. Pool callers share cores with workers. One socket
uses local NUMA memory; two sockets interleave memory across their local nodes.
The table is shared, and the two-socket pool is one global `SimpleThreadPool`.

## Choose lookup counts from a pre-run

```bash
python3 benchmarks/host_layer/run.py --plan-only
bench_pre=$(mktemp -d /tmp/hostlayer-pre-XXXXXX)
python3 benchmarks/host_layer/run.py --steps 1000 --repeats 1 --out "$bench_pre"
```

`--steps` is the number of lookups **per caller, in each measured phase**.
Read `inference_time_s` and `latency_time_s` in the summary. Choose counts manually
for each case: if 1,000 steps take 0.5 seconds, approximately 20,000 steps target
10 seconds. Use the faster phase when sizing both phases; check the resulting
times. No automatic calibration or duration-based stopping is used.

```bash
bench_out=$(mktemp -d /tmp/hostlayer-results-XXXXXX)
# Example count: replace after the pre-run. Repeat for the other case names.
python3 benchmarks/host_layer/run.py --case socket-inline-full \
  --steps 20000 --repeats 5 --out "$bench_out"
# Or run all six cases with one chosen count by omitting --case.
python3 benchmarks/host_layer/run.py --report "$bench_out"
```

Reuse `--out` for different cases; an existing case is rejected to prevent mixing
measurements. Each directory contains environment/topology, command logs, native
CSVs and `summary.csv`. Paste the summary and environment before releasing the
allocation: `/tmp` is ephemeral. No separate single-core performance case is run.

## Measurement notes

- The runner uses 100 warmup lookups/caller before throughput (`--warmup` overrides it). Table setup,
  key generation, verification and file writes are outside the measured phases.
- A bounded 512-batch key ring is generated before each phase and replayed
  cyclically. This preserves the previous large-table suite's memory bound;
  increasing steps does not allocate additional key batches. The original
  GPU/LinearUVM/Hierarchical harness instead stores keys for every measured step.
- Callers synchronize at phase boundaries. With fixed counts, faster callers
  can finish earlier; latency describes this closed-loop run, not fixed offered QPS.
- MKPS counts millions of requested keys/second. GB/s is logical embedding
  payload bandwidth, not measured DRAM bandwidth. Mean latency is per lookup call.
- Correctness checks cover gathered values, missing keys and final outputs.
  Host hit rate measures table hits, not CPU cache hits. Host THP policy is recorded
  and left unchanged. Earlier timed results remain reproducible at commit `0d061ca`
  with `--distributions powerlaw`; do not mix them with this fixed-count round.
