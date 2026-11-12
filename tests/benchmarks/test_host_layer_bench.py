#!/usr/bin/python
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration tests against the actual built NVE benchmark; no mock kernel."""
import csv
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

def exercise_benchmark(binary):
    binary = Path(binary).resolve()
    base = ['-nr', '257', '-ed', '7', '-b', '3', '--hotness', '65', '-ns', '4']

    def run(args, ok=True):
        p = subprocess.run([str(binary), *args], capture_output=True, text=True, timeout=60)
        assert (p.returncode == 0) == ok, (args, p.returncode, p.stdout, p.stderr)
        if ok:
            assert 'Verification: passed' in p.stdout, p.stdout
        return p

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'metrics.csv'
        for dtype in ['float32', 'float16']:
            for inline in [False, True]:
                run([*base, '-nw', '2', '-dt', dtype, '-nl', '2', '-t', '2', '--threadpool_workers', '2',
                     '--task_size_threshold', '32', '--alpha', '1', '--latency_steps', '5',
                     '--tag', 'comma,"quote"\nnewline', '-csv', str(path),
                     *(['--disable_threadpool'] if inline else [])])
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4
        for r in rows:
            assert r['device'] == 'cpu' and r['layer'] == 'host'
            assert r['verification'] == 'passed'
            assert r['tag'] == 'comma,"quote"\nnewline'
            assert int(r['lat_steps']) == 20
            assert int(r['host_hitrate_samples']) == 16
            assert float(r['host_hitrate_mean']) == 1
            assert float(r['lat_host_hitrate_mean']) == 1
            assert 0 < float(r['lat_p50_ms']) <= float(r['lat_p95_ms']) <= float(r['lat_p99_ms']) <= float(r['lat_max_ms'])
            kps = 4 * 4 * 3 * 65 / float(r['inference_time_s'])
            assert math.isclose(float(r['mkps']), kps / 1e6, rel_tol=1e-9)
            rowbytes = 7 * (4 if r['data_type'] == 'float32' else 2)
            assert math.isclose(float(r['gbps']), kps * rowbytes / 1e9, rel_tol=1e-9)
            assert math.isclose(float(r['gbps']) / float(r['gibps']), 2**30 / 1e9, rel_tol=1e-9)
        # Replay can request more latency steps than throughput steps.
        run([*base, '-a', '0', '-nw', '0', '--latency_keys', 'replay', '--latency_steps', '9'])
        skipped = Path(tmp) / 'skip.csv'
        run([*base, '--latency_steps', '0', '--hitrate_interval', '0', '-csv', str(skipped)])
        with skipped.open() as f:
            r = next(csv.DictReader(f))
        assert r['lat_steps'] == '0' and r['lat_p99_ms'] == ''
        assert r['host_hitrate_samples'] == '0' and r['host_hitrate_mean'] == ''
        # Edge sizes, uniform/strongly skewed workloads and a single-row table.
        run(['-nr', '1', '-ed', '1', '-b', '1', '--hotness', '1', '-ns', '4', '-nw', '0', '-a', '2'])
        run([*base, '-a', '-1', '--latency_keys', 'replay'])
        # Fixed counts, a two-batch ring and pinned callers/workers exercise
        # wraparound in both phases without allocating keys for every lookup.
        allowed = sorted(os.sched_getaffinity(0))
        caller_cpus = allowed[:min(2, len(allowed))]
        for inline in (False, True):
            bounded = Path(tmp) / f'bounded-{inline}.csv'
            run([*base[:-2], '-ns', '7', '--latency_steps', '5', '-t', str(len(caller_cpus)), '-nw', '1',
                 '--key_buffer_steps', '2', '--alpha', '1.15',
                 '--caller_cpus', ','.join(map(str, caller_cpus)),
                 '--threadpool_workers', str(len(caller_cpus)),
                 '--task_size_threshold', '32', '-csv', str(bounded),
                 *(['--disable_threadpool'] if inline else
                   ['--worker_cpus', ','.join(map(str, caller_cpus))])])
            with bounded.open() as f:
                r = next(csv.DictReader(f))
            assert float(r['inference_time_s']) > 0 and float(r['latency_time_s']) > 0
            assert int(r['key_bytes_per_caller']) == 2 * 3 * 65 * 8
            assert int(r['lat_steps']) == 5 * len(caller_cpus)
            assert int(r['host_hitrate_samples']) == 7 * len(caller_cpus)
            assert math.isclose(float(r['mkps']), 7 * len(caller_cpus) * 195 / float(r['inference_time_s']) / 1e6, rel_tol=1e-9)
            assert r['caller_cpus'] == ','.join(map(str, caller_cpus))
            assert float(r['lat_p99_ms']) <= float(r['lat_max_ms'])
        for args in [['-nr', '0'], ['-b', '-1'], ['--latency_steps', '-2'],
                     ['--threadpool_workers', '-1'], ['--alpha', 'nan'], ['--alpha', 'inf'],
                     ['--latency_keys', 'invalid'], ['-dt', 'invalid'], ['-ns', '0'],
                     ['--task_size_threshold', '0'], ['--hitrate_interval', '-1'],
                     ['-ed', '9223372036854775807'], ['--alpha', '1000000'],
                     ['--device', 'cuda'], ['--layer', 'hierarchical'],
                     ['--throughput_seconds', '1'], ['--warmup_seconds', '1'],
                     ['--key_buffer_steps', '0'], ['--caller_cpus', '0,0'],
                     ['--caller_cpus', '0,'], ['--worker_cpus', '0', '--disable_threadpool']]:
            run(args, ok=False)
        path.write_text('different,header\n')
        run([*base, '-csv', str(path)], ok=False)
    print('PASS: real NVE gather/default rows/final outputs; fp32/fp16; inline/pool; concurrent layers; fixed counts/bounded keys/CPU affinity; metrics/CSV; invalid arguments')

def test_host_layer_benchmark():
    import os
    import pytest
    binary = Path(os.environ.get("NVE_HOST_LAYER_BENCH_PATH", "build-host-layer/bin/host_layer_bench"))
    if not binary.is_file():
        pytest.skip("build host_layer_bench or set NVE_HOST_LAYER_BENCH_PATH")
    exercise_benchmark(binary)


if __name__ == "__main__":
    exercise_benchmark(sys.argv[1] if len(sys.argv) > 1 else "build-host-layer/bin/host_layer_bench")
