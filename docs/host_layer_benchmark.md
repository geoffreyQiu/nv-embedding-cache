# HostLayer CPU inference benchmark

`host_layer_bench` mirrors `single_layer_bench.cpp` for the CPU lookup path:
`HostEmbeddingLayer<int64_t>` over a dense `LinearHostTable<int64_t>` with ordinary
host buffers. It measures the C++ lookup underlying Python
`LayerType.HostLayer, device="cpu"`. Python dispatch and PyTorch tensor allocation
are excluded. The benchmark performs no training, pooling, GPU gather, or
parameter-server access.

## Build

From the repository root, in a CUDA development environment:

```bash
BUILD_JOBS=4 bash benchmarks/host_layer/build.sh
```

The helper initializes required submodules and builds the native CMake target
`host_layer_bench` in `build-host-layer/`. It selects Release, driverless mode,
no MPI, no Torch bindings, no plugins, and no upstream tests/samples.
`NVE_BUILD_HOST_BENCHMARK=ON` keeps this benchmark enabled in that minimal build.
The target is also included in a normal build with tests/samples enabled.
Extra CMake arguments can be appended, such as `-DNVE_DISABLE_AVX512=ON`.

CUDA toolkit/runtime libraries are still required to build and link NVE;
driverless means no CUDA driver is required to execute these CPU lookups.
The SDK may query driver availability and pointer attributes internally. The
harness issues no CUDA device, allocation, copy, or stream-synchronization calls.
Upstream still compiles its generated CUDA kernels into the shared library, so
the first build takes several minutes. Rebuild for a different CPU instruction set.

For the provided NVIDIA PyTorch 26.03 container:

```bash
BUILD_JOBS=8 bash benchmarks/host_layer/docker-build.sh
bash benchmarks/host_layer/test.sh
bash benchmarks/host_layer/docker-run.sh --help
```

On managed Docker hosts requiring an explicit runtime, prefix commands with
`DOCKER_RUNTIME=nvidia`. The image defaults to `nve-host-layer-bench:26.03`;
`HOSTLAYER_IMAGE` overrides it. No GPU allocation or `--gpus` flag is needed.
`NVIDIA_VISIBLE_DEVICES=void` and `CUDA_VISIBLE_DEVICES=-1` disable CUDA device
visibility. Some managed runtimes still inject NVIDIA device nodes;
`collect-env.sh` inventories those nodes and checks that no CUDA device is usable.

## Run

```bash
bash benchmarks/host_layer/docker-run.sh \
  --num_table_rows 1000000 --embedding_dim 128 --data_type float32 \
  --batch 32 --hotness 512 --alpha 1.05 \
  --num_warmup_steps 100 --num_steps 100 \
  --threads 1 --disable_threadpool \
  --csv_filename /results/hostlayer.csv
```

The container contains the source, shared library, and executable. The run helper
copies results to `build-host-layer/results/` using Docker archive operations,
which also work with remote daemons that cannot bind-mount the source checkout.
Existing CSVs are copied into each container for sequential appends. Run the
helper sequentially for a given checkout to avoid competing result copies.
Only files written under `/results` are extracted.

For a native run, use `build-host-layer/bin/host_layer_bench` directly. The
standalone integration suite works with either the helper or another build:

```bash
python3 tests/benchmarks/test_host_layer_bench.py build-host-layer/bin/host_layer_bench
# Optional pytest discovery:
NVE_HOST_LAYER_BENCH_PATH=build-host-layer/bin/host_layer_bench \
  python3 -m pytest tests/benchmarks/test_host_layer_bench.py
```

The tests exercise byte-exact fp32/fp16 gather, default rows, final measured
outputs, odd row widths, multiple callers/layers, worker-pool and inline modes,
fresh/replayed keys, metric arithmetic, CSV quoting/appending, and invalid inputs.

## Thread and workload controls

| Flag | Meaning |
|---|---|
| `--threads/-t` | Concurrent inference callers per layer; default 1 |
| `--num_layers/-nl` | Independent full tables; default 1 |
| `--threadpool_workers` | Shared NVE workers across all layers/callers; default 1, 0 uses the library default |
| `--disable_threadpool` | Execute each lookup on its caller thread using InlineThreadPool |
| `--task_size_threshold` | Keys per task threshold for table/layer; default 1024 |
| `--alpha/-a` | Power-law exponent, default 1.05; values <= 0 select uniform input |
| `--seed` | RNG/permutation seed; default 42 |
| `--latency_steps` | Steps per caller; -1 means num_steps, 0 skips the phase |
| `--latency_keys` | `fresh` (default) or cyclic `replay` of throughput keys |
| `--hitrate_interval` | Sample every N steps; default 1, 0 disables sampling |
| `--max_latency_ms` | Count latency samples above a positive budget |

`batch * hotness` is the number of independent keys per lookup; hotness does not
request embedding-bag reduction. Keys are valid dense row IDs, so host hit rate
should be 1. It measures table hits, not CPU hardware-cache hits.

The generator matches single_layer_bench: a discretized continuous power-law
inverse CDF followed by a seed-based affine permutation. This is not an exact
discrete Zipf sampler. Alpha near 1 is nudged to 1+1e-6; unsupported numerical
ranges and nonfinite values are rejected.

Example sequential caller-scaling sweep:

```bash
ALPHAS='0 0.8 1.05 1.2' CALLERS='1 2 4 8' REPEATS=3 \
  ROWS=1000000 DIM=128 BATCH=64 HOTNESS=256 STEPS=100 WARMUP=100 \
  bash benchmarks/host_layer/sweep.sh
```

This uses inline execution and appends `build-host-layer/results/sweep.csv`.
For worker-pool scaling, keep caller count fixed and vary worker count.

## Measurement boundaries

Tables are initialized with deterministic byte payloads and first-touched before
timing. These are opaque fp32/fp16 bit patterns: no arithmetic or conversion is
performed. Preflight verifies boundary rows, missing-key zero defaults and hit
rate. Every caller verifies its entire final measured output outside timing.
There are no inserts or updates in timed phases; LinearHostTable population is
a direct write to its backing array.

Keys are generated outside each timed phase. Requested warmup uses fresh keys;
one additional untimed lookup always first-touches output/scratch, including
when the configured warmup count is zero. Callers synchronize at phase boundaries.
Throughput is total requested keys divided by the wall-clock window from the
earliest caller start to the latest finish. CPU lookup completes synchronously.

All throughput callers finish before fresh latency keys are generated. Fresh
keys still share the existing hardware-cache state. Latencies combine per-lookup
wall times across callers/layers, including worker-pool queueing. Sample storage
is reserved before timing. Percentiles use linear interpolation between sorted
samples. Per-layer throughput is printed separately.

CSV records workload/thread settings, revision, MKPS, throughput seconds, host
hit statistics, latency mean/p50/p95/p99/max, budget violations, setup/prefill time,
and peak RSS. An incompatible CSV header is rejected. Source revision can be
supplied with `NVE_BENCH_SOURCE_REV` in CMake; otherwise it is detected from Git
or reported as unknown in source archives. The Docker build helper marks a
working tree with uncommitted files as dirty.

Bandwidth counts useful embedding payload bytes, not measured DRAM traffic.
Cache reuse, row reads, output writes, and cache-line amplification are not
separately accounted. `gbps` is decimal GB/s; `gibps` uses 2^30 bytes, matching
the original benchmark's bandwidth formula despite its GB/s label.

Approximate memory cost:

```text
tables = num_layers * num_table_rows * embedding_dim * dtype_bytes
per caller keys = max(num_steps, latency_steps) * batch * hotness * 8  # fresh
per caller output = batch * hotness * embedding_dim * dtype_bytes
per caller warmup keys = batch * hotness * 8
callers = num_layers * threads
```

Replay allocates only num_steps key sets. Add hitmask/scratch and runtime memory.
Defaults require about 488 MiB of table memory plus 200 MiB of keys and 128 MiB
of output per caller. Reduce batch/hotness/steps when increasing caller count.
Arithmetic overflow is checked, but there is no automatic RAM cap.

`DOCKER_CPUSET`, `DOCKER_MEMSET`, and `DOCKER_CPUS` pass daemon CPU/memory limits.
For native runs, numactl can bind CPUs and memory nodes. Initialization is serial,
so control NUMA placement explicitly. Record CPU topology, frequency policy,
compiler, image digest, and affinity settings for controlled comparisons:

```bash
bash benchmarks/host_layer/collect-env.sh
```

This saves `build-host-layer/results/environment.txt` for the actual Docker host.
Its topology may differ from the client. Record run-specific affinity limits
separately; the collector inspects an unrestricted container.
