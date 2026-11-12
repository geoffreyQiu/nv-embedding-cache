#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the three HostLayer CPU scenarios with explicit lookup counts."""
import argparse
import csv
import os
import platform
import random
import shlex
import statistics
import subprocess
import tempfile
from pathlib import Path

CASES = [f"{scope}-{kind}" for scope in ("socket", "server")
         for kind in ("inline-full", "inline-64", "pool-16")]
METRICS = ("mkps", "gbps", "lat_mean_ms", "inference_time_s", "latency_time_s")


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return "unavailable"


def command(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def numbers(text):
    result = set()
    for part in text.split(","):
        limits = list(map(int, part.split("-")))
        result.update(range(limits[0], limits[-1] + 1))
    return result


def spread(cpus, count):
    if not 1 <= count <= len(cpus):
        raise ValueError("Not enough accessible physical cores")
    return [cpus[0]] if count == 1 else [cpus[i * (len(cpus) - 1) // (count - 1)] for i in range(count)]


def make_cases(rows, allowed, memory_nodes, smoke=False):
    # Socket/core IDs are topology labels, not necessarily consecutive numbers.
    cores = {}
    for cpu, core, socket, node in sorted(rows):
        if cpu in allowed:
            cores.setdefault((socket, core), (cpu, node))
    sockets = sorted({s for s, c in cores})
    if not sockets or (not smoke and (len(sockets) != 2 or set(cores) != {(r[2], r[1]) for r in rows})):
        raise ValueError("Expose both sockets and all physical cores; use an exclusive node and --cpu-bind=none")
    by_socket = {s: sorted(cpu for (socket, core), (cpu, node) in cores.items() if socket == s) for s in sockets}
    if smoke:
        by_socket = {s: cpus[:2] for s, cpus in by_socket.items()}
    elif len({len(v) for v in by_socket.values()}) != 1 or min(map(len, by_socket.values())) < 64:
        raise ValueError("Expected two equal sockets with at least 64 physical cores each")
    node_for = {cpu: node for cpu, node in cores.values()}
    cases = {}
    for scope, selected in (("socket", sockets[:1]), ("server", sockets)):
        if scope == "server" and len(sockets) != 2:
            continue
        for kind in ("inline-full", "inline-64", "pool-16"):
            count = len(by_socket[selected[0]]) if kind == "inline-full" else (64 if kind == "inline-64" else 16)
            if smoke:
                count = min(count, *(len(by_socket[s]) for s in selected))
            callers = [cpu for s in selected for cpu in spread(by_socket[s], count)]
            workers = [cpu for s in selected for cpu in by_socket[s]] if kind == "pool-16" else []
            cpus = sorted(set(callers + workers))
            nodes = sorted({node_for[cpu] for cpu in cpus})
            if not set(nodes) <= memory_nodes:
                raise ValueError("Selected CPUs have unavailable or memoryless local NUMA nodes")
            cases[f"{scope}-{kind}"] = dict(sockets=len(selected), callers=callers, workers=workers,
                cpus=cpus, nodes=nodes, policy="membind" if len(nodes) == 1 else "interleave")
    return cases


def bench_command(binary, case, args, csv_path):
    cmd = ["numactl", "--physcpubind=" + ",".join(map(str, case["cpus"])),
        "--" + case["policy"] + "=" + ",".join(map(str, case["nodes"])), str(binary),
        "--num_table_rows", str(args.rows), "--embedding_dim", "128", "--data_type", "float32",
        "--batch", "32", "--hotness", "512", "--alpha", "1.15", "--seed", "42",
        "--num_steps", str(args.steps), "--latency_steps", str(args.steps),
        "--num_warmup_steps", str(args.warmup), "--key_buffer_steps", "512",
        "--threads", str(len(case["callers"])), "--caller_cpus", ",".join(map(str, case["callers"])),
        "--csv_filename", str(csv_path)]
    if case["workers"]:
        cmd += ["--threadpool_workers", str(len(case["workers"])),
                "--worker_cpus", ",".join(map(str, case["workers"]))]
    else:
        cmd += ["--disable_threadpool"]
    return cmd


def environment(revision):
    lines = [f"Revision: {revision}", f"Server: {platform.node()}", f"Kernel: {platform.release()}",
        f"Container OS: {platform.freedesktop_os_release().get('PRETTY_NAME')}",
        f"glibc: {' '.join(platform.libc_ver())}",
        f"NGC release: {os.environ.get('NVIDIA_PYTORCH_VERSION', 'unreported')}",
        f"Base page bytes: {os.sysconf('SC_PAGE_SIZE')}"]
    for label, cmd in (("CPU", ["lscpu"]), ("NUMA", ["numactl", "--hardware"]),
        ("Compiler", ["c++", "--version"]), ("CUDA toolkit", ["nvcc", "--version"]),
        ("PyTorch", ["python3", "-c", "import torch; print(torch.__version__, 'CUDA', torch.version.cuda)"])):
        try:
            lines.append(label + ":\n" + command(cmd))
        except (OSError, subprocess.CalledProcessError) as error:
            lines.append(label + ": " + str(error))
    for field in ("scaling_driver", "scaling_governor", "scaling_min_freq", "scaling_max_freq"):
        values = sorted({read(p) for p in Path('/sys/devices/system/cpu/cpufreq').glob(f'policy*/{field}')})
        lines.append(f"{field}: {', '.join(values) or 'unavailable'}")
    for path in ('/sys/kernel/mm/transparent_hugepage/enabled',
                 '/sys/kernel/mm/transparent_hugepage/defrag', '/proc/sys/kernel/numa_balancing',
                 '/sys/fs/cgroup/cpu.max', '/sys/fs/cgroup/memory.max'):
        lines.append(f"{path}: {read(path)}")
    return "\n".join(lines) + "\n"


def validate(row, case, args, revision):
    expected = dict(verification="passed", nve_revision=revision, alpha="1.15", num_layers="1",
        num_table_rows=str(args.rows), embedding_dim="128", data_type="float32", batch="32", hotness="512",
        num_steps=str(args.steps), num_warmup_steps=str(args.warmup), threads=str(len(case['callers'])),
        lat_steps=str(args.steps * len(case['callers'])), disable_threadpool=str(int(not case['workers'])),
        caller_cpus=','.join(map(str, case['callers'])), worker_cpus=','.join(map(str, case['workers'])))
    if case['workers']:
        expected['threadpool_workers_effective'] = str(len(case['workers']))
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"Unexpected {key}: {row.get(key)!r}, expected {value!r}")


def report(out):
    summary = []
    for name in CASES:
        path = out / f"{name}.csv"
        if not path.exists():
            continue
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        if not rows or any(r['verification'] != 'passed' for r in rows):
            raise ValueError(f"Invalid results: {path}")
        first = rows[0]
        row = dict(case=name, runs=len(rows), rows=first['num_table_rows'], steps_per_caller=first['num_steps'], callers=first['threads'],
            workers=0 if first['disable_threadpool'] == '1' else first['threadpool_workers_effective'])
        row.update({key: f"{statistics.mean(float(r[key]) for r in rows):.6f}" for key in METRICS})
        summary.append(row)
    if not summary:
        raise ValueError("No completed measurements in this directory")
    with (out / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    print((out / 'summary.csv').read_text(), end='')
    print('Measured phase times are wall seconds; latency is ms per 16,384-key call.')
    print('Results:', out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=['all'] + CASES, default='all')
    p.add_argument('--steps', type=int, default=1000, help='lookups per caller in EACH measured phase')
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--rows', type=int, default=33554432, help='16 GiB by default')
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--out', type=Path, help='new directory, or an existing directory with other cases')
    p.add_argument('--report', type=Path, help='print a saved directory summary without running')
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--smoke', action='store_true', help='tiny correctness check; not performance data')
    args = p.parse_args()
    if args.report:
        report(args.report.resolve()); return
    if min(args.steps, args.rows, args.repeats) < 1 or args.warmup < 0:
        p.error('steps, rows and repeats must be positive; warmup must be nonnegative')
    if args.smoke:
        args.steps, args.rows, args.warmup, args.repeats = 4, 257, 1, 1
    binary = Path('build-host-layer/bin/host_layer_bench').resolve()
    revision = command(['git', 'rev-parse', 'HEAD'])
    if not args.smoke and subprocess.run(['git', 'diff', '--quiet', 'HEAD']).returncode:
        p.error('Use a committed checkout and rebuild before comparing servers')
    topology = command(['lscpu', '--online', '-p=CPU,CORE,SOCKET,NODE'])
    rows = [tuple(map(int, line.split(','))) for line in topology.splitlines() if line and not line.startswith('#')]
    status = dict(line.split(':', 1) for line in read('/proc/self/status').splitlines() if ':' in line)
    allowed_nodes = numbers(status['Mems_allowed_list'].strip())
    memory_nodes = {int(path.name[4:]) for path in Path('/sys/devices/system/node').glob('node[0-9]*')
        if int(path.name[4:]) in allowed_nodes and
        any('MemTotal:' in line and int(line.split()[-2]) > 0 for line in read(path/'meminfo').splitlines())}
    cases = make_cases(rows, os.sched_getaffinity(0), memory_nodes, args.smoke)
    if args.case != 'all':
        if args.case not in cases:
            p.error('The selected case needs two accessible sockets')
        cases = {args.case: cases[args.case]}
    for name, case in cases.items():
        print(f"{name}: callers={len(case['callers'])}, workers={len(case['workers'])}, "
              f"physical cores={len(case['cpus'])}, {case['policy']}={case['nodes']}")
    print(f'Power law alpha=1.15; table={args.rows*512/2**30:.3f} GiB; '
          f'{args.steps} lookups/caller/phase; repeats={args.repeats}; SMT siblings unused.')
    if args.plan_only:
        return
    if not binary.is_file():
        p.error('Build host_layer_bench first')
    if command([str(binary), '--version']).split()[-1] != revision:
        p.error('Binary revision differs from checkout; rebuild first')
    root = Path('build-host-layer/results'); root.mkdir(parents=True, exist_ok=True)
    out = (args.out or Path(tempfile.mkdtemp(prefix='hostlayer-', dir=root))).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env_path = out/'environment.txt'
    if env_path.exists() and env_path.read_text().splitlines()[0] != f'Revision: {revision}':
        p.error('Output directory belongs to a different source revision')
    for name in cases:
        if (out/f'{name}.csv').exists():
            p.error(f'{name} already has results; choose a new output directory')
    if not env_path.exists():
        env_path.write_text(environment(revision))
        (out/'topology.csv').write_text(topology+'\n')
    rng = random.Random(2026)
    for repeat in range(1, args.repeats+1):
        order = list(cases); rng.shuffle(order)
        for name in order:
            path = out/f'{name}.csv'
            cmd = bench_command(binary, cases[name], args, path)
            print(f'Round {repeat}/{args.repeats}: {name}', flush=True)
            with (out/f'{name}-r{repeat}.log').open('w') as log:
                log.write(shlex.join(cmd)+'\n'); log.flush()
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': '-1'})
            with path.open(newline='') as f:
                results = list(csv.DictReader(f))
            if len(results) != repeat:
                raise ValueError(f'Unexpected result count: {path}')
            validate(results[-1], cases[name], args, revision)
    report(out)


if __name__ == '__main__':
    os.environ['LC_ALL'] = 'C'
    main()
