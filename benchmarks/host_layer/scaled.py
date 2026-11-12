#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Large shared-table, duration-based HostLayer socket/server comparison."""
import argparse
import collections
import csv
import json
import math
import os
import platform
import random
import re
import shlex
import statistics
import subprocess
import tempfile
from pathlib import Path

from multicore import command, numbers, read, verify_binary

FIELDS = ["mkps", "gbps", "lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "lat_p99_ms"]
ALPHAS = {"powerlaw": 1.15, "uniform": 0.0}


def spread(cpus, count):
    if not 1 <= count <= len(cpus):
        raise ValueError("Requested callers exceed available physical cores per socket")
    return [cpus[0]] if count == 1 else [cpus[i * (len(cpus) - 1) // (count - 1)] for i in range(count)]


def make_plan(rows, allowed, memory_nodes, smoke=False):
    physical = {}
    for cpu, core, socket, node in sorted(rows):
        if cpu in allowed:
            physical.setdefault((socket, core), (cpu, node))
    if not physical:
        raise ValueError("No accessible physical cores")
    if not smoke and set(physical) != {(r[2], r[1]) for r in rows}:
        raise ValueError("Allocation does not expose all physical cores; use an exclusive node and --cpu-bind=none")
    sockets = sorted({socket for socket, core in physical})
    if not smoke and len(sockets) != 2:
        raise ValueError("This comparison requires exactly two CPU sockets")
    by_socket = {s: sorted(cpu for (socket, core), (cpu, node) in physical.items() if socket == s) for s in sockets}
    if not smoke and (min(map(len, by_socket.values())) < 64 or len({len(v) for v in by_socket.values()}) != 1):
        raise ValueError("Expected two equal sockets with at least 64 physical cores each")
    cpu_node = {cpu: node for cpu, node in physical.values()}
    cases = {}
    scopes = [("socket", sockets[:1]), ("server", sockets)]
    if smoke:
        scopes = [("smoke", sockets[:1])]
        if len(sockets) == 2:
            scopes.append(("smoke-server", sockets))
        by_socket = {s: cpus[:min(2, len(cpus))] for s, cpus in by_socket.items()}
    for scope, selected_sockets in scopes:
        p = len(by_socket[selected_sockets[0]])
        if smoke:
            p = min(len(by_socket[s]) for s in selected_sockets)
            # The smoke case needs equal tiny groups even under a partial CPU allocation.
            for s in selected_sockets:
                by_socket[s] = by_socket[s][:p]
        scenarios = [("inline-full", p, False), ("inline-64", 64, False)]
        scenarios += [(f"pool-{c}", c, True) for c in (16, 32, 64)]
        if smoke:
            scenarios = [("inline", p, False), ("pool", p, True)]
        for scenario, callers_per_socket, pool in scenarios:
            callers = [cpu for s in selected_sockets for cpu in spread(by_socket[s], callers_per_socket)]
            workers = [cpu for s in selected_sockets for cpu in by_socket[s]] if pool else []
            cpus = sorted(set(callers + workers))
            nodes = sorted({cpu_node[cpu] for cpu in cpus})
            if not set(nodes) <= memory_nodes:
                raise ValueError("Selected CPUs have unavailable or memoryless local NUMA nodes; inspect topology")
            cases[f"{scope}-{scenario}"] = dict(scope=scope, sockets=len(selected_sockets),
                scenario=scenario, callers_per_socket=callers_per_socket,
                physical_cores_per_socket=p, callers=len(callers), workers=len(workers),
                caller_cpus=callers, worker_cpus=workers, cpus=cpus, nodes=nodes,
                memory_policy="membind" if len(nodes) == 1 else "interleave")
    return cases


def cache_size(text):
    match = re.fullmatch(r"(\d+)\s*([KMG]?)", text.strip(), re.I)
    if not match:
        return None
    return int(match[1]) * {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[match[2].upper()]


def l3_info(rows):
    instances = {}
    for cpu, core, socket, node in rows:
        for path in Path(f"/sys/devices/system/cpu/cpu{cpu}/cache").glob("index*"):
            if read(path / "level") != "3":
                continue
            shared = read(path / "shared_cpu_list")
            size = cache_size(read(path / "size"))
            if size is not None and shared != "unavailable":
                instances[(socket, shared, read(path / "type"))] = size
    return {str(s): sum(size for (socket, shared, kind), size in instances.items() if socket == s)
            for s in sorted({key[0] for key in instances})}


def environment(rows, revision):
    info = dict(server=platform.node(), kernel=platform.release(),
        container_os=platform.freedesktop_os_release().get("PRETTY_NAME"),
        glibc=" ".join(platform.libc_ver()), source=revision,
        ngc_release=os.environ.get("NVIDIA_PYTORCH_VERSION", "unreported"),
        base_page_bytes=os.sysconf("SC_PAGE_SIZE"), l3_bytes_per_socket=l3_info(rows))
    lscpu = dict(line.split(":", 1) for line in command(["lscpu"]).splitlines() if ":" in line)
    for name in ("Architecture", "Model name", "Socket(s)", "Core(s) per socket", "Thread(s) per core", "CPU(s)", "NUMA node(s)"):
        info[name] = lscpu.get(name, "unavailable").strip()
    for label, args in (
        ("compiler", ["c++", "--version"]), ("cuda_toolkit", ["nvcc", "--version"]),
        ("pytorch", ["python3", "-c", "import torch; print(torch.__version__); print('CUDA', torch.version.cuda)"])):
        try:
            output = command(args).splitlines()
            info[label] = next((s for s in output if "compilation tools" in s), output[0]) if label != "pytorch" else "; ".join(output)
        except (OSError, subprocess.CalledProcessError, IndexError) as error:
            info[label] = str(error)
    for field in ("scaling_driver", "scaling_governor", "scaling_min_freq", "scaling_max_freq", "energy_performance_preference"):
        values = [read(p) for p in Path("/sys/devices/system/cpu/cpufreq").glob(f"policy*/{field}")]
        if values and all(v.isdigit() for v in values):
            values = list(map(int, values))
            info[field] = f"{min(values)}–{max(values)} kHz ({len(values)} policies)"
        else:
            info[field] = dict(collections.Counter(values)) or "unavailable"
    thp = Path("/sys/kernel/mm/transparent_hugepage")
    info["THP"] = {name: read(thp / name) for name in ("enabled", "defrag", "hpage_pmd_size")}
    info["THP_sizes"] = {p.parent.name: read(p) for p in thp.glob("hugepages-*kB/enabled")}
    info["numa_balancing"] = read("/proc/sys/kernel/numa_balancing")
    info["cgroup"] = {f: read(Path("/sys/fs/cgroup") / f) for f in ("cpu.max", "memory.max", "memory.current", "pids.max", "pids.current")}
    info["numa_hardware"] = command(["numactl", "--hardware"])
    return info


def check_limits(cases, table_bytes, key_steps, env):
    count = max(c["callers"] for c in cases.values())
    memory = table_bytes + count * (key_steps * 16384 * 8 + 16384 * 512 + 24_000_000)
    quota = env["cgroup"]["cpu.max"].split()
    if len(quota) == 2 and quota[0].isdigit() and int(quota[0]) / int(quota[1]) < max(len(c["cpus"]) for c in cases.values()):
        raise ValueError("Visible cgroup CPU quota is too small for this suite")
    for resource, required in (("memory", math.ceil(memory * 1.25)),
                               ("pids", max(c["callers"] + c["workers"] for c in cases.values()) + 8)):
        limit, current = (env["cgroup"][f"{resource}.{field}"] for field in ("max", "current"))
        if limit.isdigit() and current.isdigit() and int(limit) - int(current) < required:
            raise ValueError(f"Insufficient visible cgroup {resource} allowance")
    return memory / 2**30


def bench_command(binary, case, config, distribution, csv_path, tag):
    cmd = ["numactl", "--physcpubind=" + ",".join(map(str, case["cpus"])),
           f"--{case['memory_policy']}=" + ",".join(map(str, case["nodes"])), str(binary),
           "--num_layers", "1", "--num_table_rows", str(config["rows"]),
           "--embedding_dim", "128", "--data_type", "float32", "--batch", "32", "--hotness", "512",
           "--alpha", str(ALPHAS[distribution]), "--seed", "42", "--num_steps", "1", "--latency_steps", "1",
           "--num_warmup_steps", "1" if config["smoke"] else "100",
           "--throughput_seconds", str(config["seconds"]), "--latency_seconds", str(config["seconds"]),
           "--warmup_seconds", str(config["warmup_seconds"]), "--key_buffer_steps", str(config["key_steps"]),
           "--latency_keys", "fresh", "--task_size_threshold", "1024", "--hitrate_interval", "1",
           "--threads", str(case["callers"]), "--caller_cpus", ",".join(map(str, case["caller_cpus"])),
           "--csv_filename", str(csv_path), "--tag", tag]
    if case["workers"]:
        cmd += ["--threadpool_workers", str(case["workers"]), "--worker_cpus", ",".join(map(str, case["worker_cpus"]))]
    else:
        cmd += ["--disable_threadpool"]
    return cmd


def validate(row, case, config, revision, distribution):
    expected = dict(verification="passed", nve_revision=revision, timing_mode="duration",
        threads=str(case["callers"]), num_layers="1", num_table_rows=str(config["rows"]),
        table_bytes_per_layer=str(config["rows"] * 512), data_type="float32", embedding_dim="128",
        batch="32", hotness="512", task_size_threshold="1024",
        disable_threadpool=str(int(case["workers"] == 0)),
        caller_cpus=",".join(map(str, case["caller_cpus"])),
        worker_cpus=",".join(map(str, case["worker_cpus"])),
        key_buffer_steps_effective=str(config["key_steps"]))
    if case["workers"]:
        expected["threadpool_workers_effective"] = str(case["workers"])
    for field, value in expected.items():
        if row.get(field) != value:
            raise ValueError(f"Unexpected result field {field}: {row.get(field)!r}, expected {value!r}")
    if float(row["alpha"]) != ALPHAS[distribution]:
        raise ValueError("Wrong key distribution")
    if any(float(row[f]) < config["seconds"] for f in ("inference_time_s", "latency_time_s")):
        raise ValueError("Measured phase shorter than requested")
    if int(row["lat_steps"]) < case["callers"] or int(row["throughput_calls_min_per_caller"]) < 1:
        raise ValueError("A caller produced no measured requests")
    calls = int(row["throughput_completed_calls"])
    expected_mkps = calls * 16384 / float(row["inference_time_s"]) / 1e6
    if not math.isclose(float(row["mkps"]), expected_mkps, rel_tol=1e-8):
        raise ValueError("Throughput does not match completed request count and elapsed time")
    if not math.isclose(float(row["gbps"]), float(row["mkps"]) * .512, rel_tol=1e-8):
        raise ValueError("Payload bandwidth does not match key throughput")
    if not all(math.isfinite(float(row[f])) and float(row[f]) > 0 for f in FIELDS):
        raise ValueError("Non-positive or non-finite performance result")


def write_summary(out, records):
    groups = collections.defaultdict(list)
    for r in records:
        groups[(r["distribution"], r["case"])].append(r)
    summary = []
    for (distribution, case), rows in sorted(groups.items()):
        first = rows[0]
        record = {key: first[key] for key in ("distribution", "case", "sockets", "callers_per_socket", "callers", "workers")}
        record["runs"] = len(rows)
        record.update({f: f"{statistics.mean(float(r[f]) for r in rows):.6f}" for f in FIELDS})
        record["min_throughput_s"] = f"{min(float(r['inference_time_s']) for r in rows):.6f}"
        record["min_latency_s"] = f"{min(float(r['latency_time_s']) for r in rows):.6f}"
        record["min_latency_samples"] = min(int(r["lat_steps"]) for r in rows)
        huge = max(int(r[f]) for r in rows for f in ("throughput_anon_huge_kib", "latency_anon_huge_kib"))
        record["max_anon_huge_mib"] = f"{huge / 1024:.3f}" if huge >= 0 else "unavailable"
        summary.append(record)
    for name, data in (("runs.csv", records), ("summary.csv", summary)):
        if data:
            with (out / name).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0]))
                writer.writeheader(); writer.writerows(data)


def report(out):
    plan = json.loads((out / "plan.json").read_text())
    env = json.loads((out / "environment.json").read_text())
    lines = [f"Server: {env['server']}", f"Source: {plan['revision']}",
             f"Status: {plan['status']}; completed {plan.get('completed_runs', 0)}/{plan['expected_runs']} runs"]
    for key in ("Model name", "Architecture", "Socket(s)", "Core(s) per socket", "Thread(s) per core", "CPU(s)", "NUMA node(s)", "kernel", "container_os", "glibc",
                "ngc_release", "pytorch", "compiler", "cuda_toolkit", "base_page_bytes", "l3_bytes_per_socket",
                "scaling_driver", "scaling_governor", "scaling_min_freq", "scaling_max_freq", "energy_performance_preference", "numa_balancing", "THP", "cgroup"):
        lines.append(f"{key}: {env[key]}")
    lines += [line for line in env["numa_hardware"].splitlines() if line.startswith("available:") or re.match(r"node \d+ size:", line)]
    lines.append("THP size policies (except never): " + str({k: v for k, v in env["THP_sizes"].items() if "[never]" not in v}))
    lines += [f"Workload: {plan['config']}", "One shared table per process; server = two sockets; SMT unused.",
              "Mean/p99 are per lookup call (16384 keys), under closed-loop concurrency.",
              "Summary percentiles average per-run percentiles; compare matching distribution/scope/caller count.",
              "Full-core latency uses different caller counts on different products.",
              "AnonHugePages is process-wide, sampled after each phase outside timing."]
    summary = out / "summary.csv"
    lines += ["", summary.read_text() if summary.exists() else "No completed measurements."]
    runs_path = out / "runs.csv"
    if runs_path.exists():
        groups = collections.defaultdict(list)
        with runs_path.open(newline="") as f:
            for r in csv.DictReader(f):
                groups[(r["distribution"], r["case"])].append(float(r["mkps"]))
        lines.append("Per-run throughput (MKPS):")
        lines += [f"{d}/{c}: " + ", ".join(f"{v:.2f}" for v in values) for (d, c), values in sorted(groups.items())]
    text = "\n".join(lines) + "\n"
    (out / "results-for-chat.txt").write_text(text)
    print(text, end="")
    print("Results directory:", out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--table-gib", type=int, default=16)
    p.add_argument("--distributions", nargs="+", choices=list(ALPHAS), default=list(ALPHAS))
    p.add_argument("--smoke", action="store_true", help="tiny real-NVE validation; not performance data")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--report", type=Path, help="print a saved run's environment and results; use 'latest' for the latest full run")
    args = p.parse_args()
    if args.report:
        location = args.report
        if str(location) == "latest":
            pointer = Path("build-host-layer/results/latest-scaled.txt")
            if not pointer.is_file():
                p.error("No full scaled run has started here; use the printed directory for a smoke/plan report")
            location = Path(pointer.read_text().strip())
        report(location.resolve()); return
    if not math.isfinite(args.seconds) or not 5 <= args.seconds <= 3600 or args.repeats < 1 or args.table_gib <= 8:
        p.error("full runs require 5–3600 seconds per phase, positive repeats and a table larger than 8 GiB")
    if len(set(args.distributions)) != len(args.distributions):
        p.error("duplicate distributions")
    binary = Path("build-host-layer/bin/host_layer_bench").resolve()
    if not binary.is_file():
        p.error("Build host_layer_bench first")
    revision = command(["git", "rev-parse", "HEAD"])
    if not args.smoke and subprocess.run(["git", "diff", "--quiet", "HEAD"]).returncode:
        p.error("tracked source has uncommitted edits; use the published commit for comparable results")
    topology = command(["lscpu", "--online", "-p=CPU,CORE,SOCKET,NODE"])
    rows = [tuple(map(int, line.split(","))) for line in topology.splitlines() if line and not line.startswith("#")]
    allowed_nodes = numbers(re.search(r"^Mems_allowed_list:\s*(.+)$", read("/proc/self/status"), re.M)[1])
    memory_nodes = set()
    for path in Path("/sys/devices/system/node").glob("node[0-9]*"):
        match = re.search(r"MemTotal:\s+(\d+)", read(path / "meminfo"))
        if match and int(match[1]) > 0 and int(path.name[4:]) in allowed_nodes:
            memory_nodes.add(int(path.name[4:]))
    cases = make_plan(rows, os.sched_getaffinity(0), memory_nodes, args.smoke)
    config = dict(rows=257 if args.smoke else args.table_gib * 2**21,
        seconds=.1 if args.smoke else args.seconds, warmup_seconds=0 if args.smoke else 2,
        key_steps=2 if args.smoke else 512, smoke=args.smoke)
    repeats = 1 if args.smoke else args.repeats
    root = Path("build-host-layer/results"); root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="scaled-smoke-" if args.smoke else "scaled-", dir=root)).resolve()
    env = environment(rows, revision)
    memory_gib = check_limits(cases, config["rows"] * 512, config["key_steps"], env)
    if not args.smoke and config["rows"] * 512 <= sum(env["l3_bytes_per_socket"].values()):
        p.error("Table does not exceed the reported aggregate L3 capacity; increase --table-gib")
    plan = dict(server=platform.node(), revision=revision, config=config, cases=cases,
        distributions=args.distributions, repeats=repeats, status="planned",
        expected_runs=len(cases) * len(args.distributions) * repeats, completed_runs=0)
    (out / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    (out / "topology.csv").write_text(topology + "\n")
    def save_plan():
        (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    save_plan()
    print("Server:", platform.node(), "| Results:", out, flush=True)
    print(f"Shared table: {config['rows'] * 512 / 2**30:.3f} GiB; per-phase seconds: {config['seconds']}; repeats: {repeats}")
    print(f"Largest main-buffer plus reserved sample estimate: {memory_gib:.2f} GiB")
    print("THP:", env["THP"]["enabled"], "| L3 bytes/socket:", env["l3_bytes_per_socket"] or "unreported")
    for name, c in cases.items():
        print(f"{name}: sockets={c['sockets']}, callers={c['callers']} ({c['callers_per_socket']}/socket), "
              f"workers={c['workers']}, physical cores={len(c['cpus'])}, {c['memory_policy']}={c['nodes']}")
    print("SMT unused; callers and pool workers explicitly pinned; ring keys replay within each phase.", flush=True)
    if args.plan_only:
        return
    verify_binary(binary, revision, out)
    plan["status"] = "running"; save_plan()
    if not args.smoke:
        (root / "latest-scaled.txt").write_text(str(out) + "\n")
    rng, records = random.Random(2026), []
    for repeat in range(1, repeats + 1):
        order = [(d, name) for d in args.distributions for name in cases]
        rng.shuffle(order)
        for distribution, name in order:
            label = f"{distribution}-{name}"
            case = cases[name]
            csv_path = out / f"{label}.csv"
            cmd = bench_command(binary, case, config, distribution, csv_path, f"{label}-r{repeat}")
            log_path = out / f"{label}-r{repeat}.log"
            print(f"Round {repeat}/{repeats}: {label}", flush=True)
            with log_path.open("w") as log:
                log.write(shlex.join(cmd) + "\n"); log.flush()
                try:
                    result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "LC_ALL": "C"},
                        timeout=3600 + 4 * config["seconds"])
                except subprocess.TimeoutExpired:
                    plan["status"] = "failed"; save_plan()
                    raise SystemExit(f"Benchmark timed out; see {log_path}")
            if result.returncode:
                plan["status"] = "failed"; save_plan()
                raise SystemExit(f"Benchmark failed; see {log_path}")
            try:
                with csv_path.open(newline="") as f:
                    data = list(csv.DictReader(f))
                if len(data) != repeat:
                    raise ValueError(f"Incorrect result count: {csv_path}")
                row = data[-1]
                validate(row, case, config, revision, distribution)
            except (ValueError, KeyError, OSError):
                plan["status"] = "failed"; save_plan()
                raise
            records.append(dict(distribution=distribution, case=name, sockets=case["sockets"],
                callers_per_socket=case["callers_per_socket"], callers=case["callers"], workers=case["workers"], repeat=repeat, **row))
            write_summary(out, records)
            plan["completed_runs"] = len(records); save_plan()
    plan["status"] = "complete"; save_plan()
    report(out)


if __name__ == "__main__":
    os.environ["LC_ALL"] = "C"
    main()
