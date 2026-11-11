#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run socket/server HostLayer experiments from the nv-embedding-cache root."""
import argparse
import collections
import csv
import json
import os
import platform
import random
import re
import shlex
import statistics
import subprocess
import tempfile
from pathlib import Path

FIELDS = ["mkps", "gbps", "lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "lat_p99_ms"]


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


def verify_binary(binary, revision, out):
    """Reject stale builds before allocating the large multicore key buffers."""
    csv_path = out / "preflight.csv"
    cmd = [str(binary), "--num_table_rows", "64", "--embedding_dim", "8",
           "--batch", "1", "--hotness", "8", "--num_steps", "1",
           "--num_warmup_steps", "1", "--latency_steps", "1", "--threads", "1",
           "--disable_threadpool", "--csv_filename", str(csv_path)]
    with (out / "preflight.log").open("w") as log:
        run = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "LC_ALL": "C"})
    if run.returncode:
        raise SystemExit(f"Benchmark preflight failed; see {out / 'preflight.log'}")
    with csv_path.open(newline="") as f:
        records = list(csv.DictReader(f))
    if len(records) != 1 or records[0].get("verification") != "passed":
        raise SystemExit(f"Invalid benchmark preflight; see {csv_path}")
    if records[0].get("nve_revision") != revision:
        raise SystemExit("Binary revision differs from the checkout. Rebuild with "
                         "BUILD_JOBS=8 bash benchmarks/host_layer/build.sh first.")


def make_cases(rows, allowed, memory_nodes, smt, mode):
    available = [r for r in rows if r[0] in allowed]
    all_cores = {(r[2], r[1]) for r in rows}
    if {(r[2], r[1]) for r in available} != all_cores:
        raise ValueError("Not all physical cores are accessible. Fix the Slurm/container CPU allocation first.")
    if smt and {r[0] for r in available} != {r[0] for r in rows}:
        raise ValueError("Not all online logical CPUs are accessible; cannot label this a full SMT test.")
    cores = {}
    for r in sorted(available):
        cores.setdefault((r[2], r[1]), r)
    socket = min(r[2] for r in available)
    physical = sorted(cores.values())
    local = [r for r in physical if r[2] == socket]
    groups = []
    if mode == "inline":
        groups.append(("single-core", local[:1]))
    groups += [(f"socket{socket}-cores", local), ("server-cores", physical)]
    if smt:
        groups += [(f"socket{socket}-smt", [r for r in available if r[2] == socket]),
                   ("server-smt", available)]
    cases = {}
    for label, selected in groups:
        nodes = sorted({r[3] for r in selected})
        if not set(nodes) <= memory_nodes:
            raise ValueError(f"{label}: a CPU node has no accessible local memory; explicit topology review required.")
        cpus = sorted(r[0] for r in selected)
        policy = "membind" if len(nodes) == 1 else "interleave"
        cases[label] = dict(cpus=cpus, nodes=nodes, memory_policy=policy,
                            callers=len(cpus), workers=0 if mode == "inline" else len(cpus))
    return cases


def make_socket_tuning_cases(rows, allowed, memory_nodes):
    """Few callers with a full socket pool, plus a 64-core inline comparison."""
    baseline = make_cases(rows, allowed, memory_nodes, False, "pool")
    name, socket = next((name, case) for name, case in baseline.items()
                        if name.startswith("socket"))
    prefix = name.removesuffix("-cores")
    cpus = socket["cpus"]
    if len(cpus) < 64:
        raise ValueError("Socket tuning requires at least 64 accessible physical cores in the first socket.")
    cases = {
        f"{prefix}-c{callers}-full-pool": dict(socket, mode="pool", callers=callers)
        for callers in (1, 2, 4)
    }
    # Spread the 64 choices across the sorted CPU list instead of taking a prefix.
    # This is a CPU-set restriction, not permanent per-thread pinning.
    selected = [cpus[i * (len(cpus) - 1) // 63] for i in range(64)]
    nodes = sorted({r[3] for r in rows if r[0] in selected})
    cases[f"{prefix}-c64-inline"] = dict(
        cpus=selected, nodes=nodes,
        memory_policy="membind" if len(nodes) == 1 else "interleave",
        callers=64, workers=0, mode="inline")
    return cases


def benchmark_command(binary, common, case, mode, label, repeat, out):
    bind = [f"--physcpubind={','.join(map(str, case['cpus']))}",
            f"--{case['memory_policy']}={','.join(map(str, case['nodes']))}"]
    threads = ["--threads", str(case["callers"])]
    threads += (["--disable_threadpool"] if mode == "inline" else
                ["--threadpool_workers", str(case["workers"])])
    return ["numactl", *bind, str(binary), *common, *threads,
            "--tag", f"{mode}-{label}-r{repeat}",
            "--csv_filename", str(out / f"{label}.csv")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["baseline", "socket-tuning"], default="baseline",
                        help="socket-tuning: 1/2/4 callers with a full socket pool; 64 callers inline")
    parser.add_argument("--mode", choices=["inline", "pool"], default="inline")
    parser.add_argument("--smt", action="store_true", help="also test all hardware threads")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.profile == "socket-tuning" and (args.mode != "inline" or args.smt):
        parser.error("socket-tuning defines its own pool/inline cases and uses one hardware thread per core; omit --mode and --smt")
    revision = command(["git", "rev-parse", "HEAD"])
    binary = Path("build-host-layer/bin/host_layer_bench").resolve()
    if not binary.is_file():
        raise SystemExit("Build host_layer_bench first.")
    topology = command(["lscpu", "--online", "-p=CPU,CORE,SOCKET,NODE"])
    rows = [tuple(map(int, line.split(","))) for line in topology.splitlines()
            if line and not line.startswith("#")]
    status = read("/proc/self/status")
    mem_allowed = numbers(re.search(r"^Mems_allowed_list:\s*(.+)$", status, re.M)[1])
    memory_nodes = set()
    for node in Path("/sys/devices/system/node").glob("node[0-9]*"):
        match = re.search(r"MemTotal:\s+(\d+)", read(node / "meminfo"))
        node_id = int(node.name[4:])
        if match and int(match[1]) > 0 and node_id in mem_allowed:
            memory_nodes.add(node_id)
    if args.profile == "socket-tuning":
        cases = make_socket_tuning_cases(rows, os.sched_getaffinity(0), memory_nodes)
    else:
        cases = make_cases(rows, os.sched_getaffinity(0), memory_nodes, args.smt, args.mode)
    mode_label = "socket-tuning" if args.profile == "socket-tuning" else args.mode
    root = Path("build-host-layer/results")
    root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=f"multicore-{mode_label}-", dir=root)).resolve()
    (out / "plan.json").write_text(json.dumps(dict(server=platform.node(),
        revision=revision, mode=mode_label, profile=args.profile,
        repeats=args.repeats, cases=cases), indent=2) + "\n")
    (out / "topology.csv").write_text(topology + "\n")
    env = [f"Server: {platform.node()}", f"Kernel: {platform.release()}",
           f"Container OS: {platform.freedesktop_os_release().get('PRETTY_NAME')}",
           f"glibc: {' '.join(platform.libc_ver())}", f"Source: {revision}",
           f"Base page size: {os.sysconf('SC_PAGE_SIZE')}",
           f"NGC release: {os.environ.get('NVIDIA_PYTORCH_VERSION', 'unreported')}",
           "Benchmark CUDA_VISIBLE_DEVICES: -1", command(["lscpu"]),
           command(["numactl", "--hardware"])]
    for tool in (["c++", "--version"], ["nvcc", "--version"],
                 ["python3", "-c", "import torch; print('PyTorch:', torch.__version__); print('PyTorch CUDA:', torch.version.cuda)"]):
        try:
            env.append(command(tool))
        except (OSError, subprocess.CalledProcessError) as error:
            env.append(f"{tool[0]}: {error}")
    for field in ("scaling_driver", "scaling_governor", "scaling_min_freq",
                  "scaling_max_freq", "energy_performance_preference"):
        values = collections.Counter(read(p) for p in
            Path("/sys/devices/system/cpu/cpufreq").glob(f"policy*/{field}"))
        env.append(f"{field}: {dict(values) or 'unavailable'}")
    thp = Path("/sys/kernel/mm/transparent_hugepage")
    paths = [thp / name for name in ("enabled", "defrag", "hpage_pmd_size")]
    paths += sorted(thp.glob("hugepages-*kB/enabled"))
    paths += [Path("/proc/sys/kernel/numa_balancing")]
    paths += [Path("/sys/fs/cgroup") / name for name in
              ("cpu.max", "memory.max", "memory.current", "pids.max", "pids.current")]
    env += [f"{p}: {read(p)}" for p in paths]
    (out / "environment.txt").write_text("\n".join(env) + "\n")
    print("Server:", platform.node(), "| Mode:", mode_label, flush=True)
    if args.profile == "socket-tuning":
        print("Kernel:", platform.release(), "| Base page size:", os.sysconf("SC_PAGE_SIZE"))
        print("Source:", revision)
    print("THP policy:", read(thp / "enabled"))
    print("Results directory:", out)
    for name, case in cases.items():
        gib = (512000000 + case["callers"] * (134217728 + 8388608)) / 2**30
        print(f"{name}: {case['callers']} callers, {case['workers']} NVE workers; "
              f"{len(case['cpus'])} selected logical CPUs; "
              f"{case['memory_policy']}={case['nodes']}; main buffers {gib:.2f} GiB")
    if args.profile == "socket-tuning":
        print("SMT unused: one hardware thread per selected physical core.")
        print("Task threshold 1024: at most 16 gather tasks per lookup; a full pool does not imply full utilization.", flush=True)
    else:
        print("Later round reminder: compare THP madvise -> always; verify actual THP usage.", flush=True)
    if args.plan_only:
        return
    # Check visible cgroup limits before starting a potentially large run.
    quota = read("/sys/fs/cgroup/cpu.max").split()
    required_cpus = max(len(c["cpus"]) for c in cases.values())
    if len(quota) == 2 and quota[0].isdigit() and int(quota[0]) / int(quota[1]) < required_cpus:
        raise SystemExit("Visible cgroup CPU quota is below the requested CPU count; fix the allocation first.")
    for resource, required in (
        ("memory", int(1.25 * (512000000 + max(c["callers"] for c in cases.values()) * 142606336))),
        ("pids", max(c["callers"] + c["workers"] for c in cases.values()) + 8),
    ):
        limit = read(f"/sys/fs/cgroup/{resource}.max")
        current = read(f"/sys/fs/cgroup/{resource}.current")
        if limit.isdigit() and current.isdigit() and int(limit) - int(current) < required:
            raise SystemExit(f"Insufficient visible cgroup {resource} allowance; see environment.txt in {out}.")
    verify_binary(binary, revision, out)
    common = """--num_table_rows 1000000 --embedding_dim 128 --batch 32 --hotness 512
        --alpha 1.15 --num_layers 1 --num_warmup_steps 100 --num_steps 1000
        --latency_steps 200 --task_size_threshold 1024""".split()
    rng = random.Random(2026)
    for repeat in range(1, args.repeats + 1):
        order = list(cases)
        rng.shuffle(order)
        for name in order:
            c = cases[name]
            mode = c.get("mode", args.mode)
            print(f"Round {repeat}/{args.repeats}: {name}", flush=True)
            cmd = benchmark_command(binary, common, c, mode, name, repeat, out)
            log_path = out / f"{name}-r{repeat}.log"
            with log_path.open("w") as log:
                log.write(shlex.join(cmd) + "\n")
                log.flush()
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "LC_ALL": "C"})
            if result.returncode:
                raise SystemExit(f"Benchmark failed; see {log_path}")
    summary = []
    per_run = []
    for name, case in cases.items():
        mode = case.get("mode", args.mode)
        with (out / f"{name}.csv").open(newline="") as f:
            records = list(csv.DictReader(f))
        if len(records) != args.repeats or any(
            r["verification"] != "passed" or r["nve_revision"] != revision
            or int(r["threads"]) != case["callers"]
            or int(r["disable_threadpool"]) != int(mode == "inline")
            or (mode == "pool" and int(r["threadpool_workers_effective"]) != case["workers"])
            for r in records
        ):
            raise SystemExit(f"Invalid results: {name}")
        per_run.append(name + ": " + ", ".join(f"{float(r['mkps']):.2f}" for r in records))
        summary.append(dict(case=name, mode=mode, runs=len(records),
            callers=case["callers"], workers=case["workers"], **{
                field: f"{statistics.mean(float(r[field]) for r in records):.6f}" for field in FIELDS}))
    summary_path = out / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print("\nFive-run averages:" if args.repeats == 5 else "\nRun averages:")
    print(summary_path.read_text(), end="")
    print("Percentiles: average of per-run percentiles across callers; latency is under concurrent load.")
    print("Saved summary:", summary_path)
    if args.profile == "socket-tuning":
        print("\nPer-run throughput (MKPS):")
        print("\n".join(per_run))
        (out / "results-for-chat.txt").write_text(
            f"Server: {platform.node()}\nSource: {revision}\n"
            f"THP: {read(thp / 'enabled')}\n"
            + summary_path.read_text() + "\nPer-run throughput (MKPS):\n"
            + "\n".join(per_run) + "\n")
    else:
        print("Later round reminder: compare THP madvise -> always; verify actual THP usage.")


if __name__ == "__main__":
    os.environ["LC_ALL"] = "C"
    main()
