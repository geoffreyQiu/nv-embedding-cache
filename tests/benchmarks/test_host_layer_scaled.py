#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Topology and metric checks for the large shared-table suite; no CUDA required."""
import csv
import importlib.util
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/host_layer"))
SPEC = importlib.util.spec_from_file_location("host_layer_scaled", ROOT / "benchmarks/host_layer/scaled.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def topology(p):
    return [(s * p + core + sibling * 2 * p, core, s, s)
            for s in range(2) for core in range(p) for sibling in range(2)]


def cases(p=128):
    rows = topology(p)
    return RUNNER.make_plan(rows, {r[0] for r in rows}, {0, 1})


class PlacementTests(unittest.TestCase):
    def test_three_products_have_ten_cases_with_per_socket_callers(self):
        for p in (88, 128, 192):
            with self.subTest(cores=p):
                plan = cases(p)
                self.assertEqual(len(plan), 10)
                for scope, sockets in (("socket", 1), ("server", 2)):
                    for kind, c, w in (("inline-full", p, 0), ("inline-64", 64, 0),
                                      ("pool-16", 16, p), ("pool-32", 32, p), ("pool-64", 64, p)):
                        case = plan[f"{scope}-{kind}"]
                        self.assertEqual((case["callers"], case["workers"]), (c * sockets, w * sockets))
                        self.assertEqual(len(set(case["caller_cpus"])), c * sockets)
                        self.assertEqual(len(set(case["worker_cpus"])), w * sockets)
                        self.assertTrue(all(cpu < p * sockets for cpu in case["cpus"]))
                        for s in range(sockets):
                            self.assertEqual(sum(s * p <= cpu < (s + 1) * p for cpu in case["caller_cpus"]), c)
                        self.assertEqual(case["nodes"], list(range(sockets)))
                        self.assertEqual(case["memory_policy"], "membind" if sockets == 1 else "interleave")

    def test_missing_core_and_inaccessible_memory_fail(self):
        rows = topology(88)
        with self.assertRaisesRegex(ValueError, "all physical cores"):
            RUNNER.make_plan(rows, set(range(1, 176)), {0, 1})
        with self.assertRaisesRegex(ValueError, "NUMA nodes"):
            RUNNER.make_plan(rows, set(range(352)), {0})

    def test_only_second_smt_sibling_accessible_is_supported(self):
        rows = topology(128)
        plan = RUNNER.make_plan(rows, set(range(256, 512)), {0, 1})
        self.assertEqual(plan["server-pool-16"]["worker_cpus"], list(range(256, 512)))

    def test_multiple_nodes_in_socket_stay_local(self):
        rows = [(cpu, core, socket, socket * 2 + core % 2) for cpu, core, socket, node in topology(128)]
        plan = RUNNER.make_plan(rows, set(range(512)), {0, 1, 2, 3})
        self.assertEqual(plan["socket-inline-full"]["nodes"], [0, 1])
        self.assertEqual(plan["server-inline-full"]["nodes"], [0, 1, 2, 3])
        self.assertEqual(plan["socket-inline-full"]["memory_policy"], "interleave")

    def test_commands_use_one_table_exact_pins_and_bounded_keys(self):
        config = dict(rows=33554432, seconds=10, warmup_seconds=2, key_steps=512, smoke=False)
        for name, case in cases(192).items():
            cmd = RUNNER.bench_command(Path("bench"), case, config, "uniform", Path("out.csv"), "test")
            for key, value in (("--num_layers", "1"), ("--num_table_rows", "33554432"),
                               ("--throughput_seconds", "10"), ("--latency_seconds", "10"),
                               ("--key_buffer_steps", "512"), ("--alpha", "0.0")):
                self.assertEqual(cmd[cmd.index(key) + 1], value)
            self.assertEqual(cmd[cmd.index("--caller_cpus") + 1], ",".join(map(str, case["caller_cpus"])))
            self.assertEqual("--disable_threadpool" in cmd, case["workers"] == 0)
            self.assertEqual("--worker_cpus" in cmd, case["workers"] > 0)

    def test_smoke_uses_only_accessible_small_cpu_set(self):
        plan = RUNNER.make_plan(topology(88), {3, 5}, {0}, smoke=True)
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan["smoke-pool"]["worker_cpus"], [3, 5])

    def test_smoke_exercises_two_socket_interleaving_when_available(self):
        plan = RUNNER.make_plan(topology(88), {3, 5, 90, 92}, {0, 1}, smoke=True)
        self.assertEqual(len(plan), 4)
        self.assertEqual(plan["smoke-server-pool"]["caller_cpus"], [3, 5, 90, 92])
        self.assertEqual(plan["smoke-server-pool"]["worker_cpus"], [3, 5, 90, 92])
        self.assertEqual(plan["smoke-server-pool"]["memory_policy"], "interleave")


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.case = cases()["server-pool-16"]
        self.config = dict(rows=33554432, seconds=10, key_steps=512)
        self.row = dict(verification="passed", nve_revision="test", timing_mode="duration", threads="32",
            num_layers="1", num_table_rows="33554432", table_bytes_per_layer=str(16 * 2**30),
            data_type="float32", embedding_dim="128", batch="32", hotness="512", task_size_threshold="1024",
            disable_threadpool="0", caller_cpus=",".join(map(str, self.case["caller_cpus"])),
            worker_cpus=",".join(map(str, self.case["worker_cpus"])), key_buffer_steps_effective="512",
            threadpool_workers_effective="256", alpha="1.15", inference_time_s="10", latency_time_s="10",
            lat_steps="1000", throughput_completed_calls="10000", throughput_calls_min_per_caller="1",
            mkps="16.384", gbps=str(16.384 * .512), lat_mean_ms="1", lat_p50_ms=".8", lat_p95_ms="1.2",
            lat_p99_ms="1.4", throughput_anon_huge_kib="0", latency_anon_huge_kib="0")

    def test_valid_metrics(self):
        RUNNER.validate(self.row, self.case, self.config, "test", "powerlaw")

    def test_bad_duration_affinity_workload_or_metrics_fail(self):
        for field, value in (("inference_time_s", ".3"), ("latency_time_s", ".2"),
                             ("caller_cpus", "0"), ("num_layers", "2"), ("num_table_rows", "1000000"),
                             ("mkps", "99"), ("gbps", "99"), ("nve_revision", "old"),
                             ("lat_steps", "0"), ("alpha", "0")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                RUNNER.validate(dict(self.row, **{field: value}), self.case, self.config, "test", "powerlaw")

    def test_summary_keeps_workloads_separate_and_averages_run_percentiles(self):
        rows = [dict(distribution=d, case="server-pool-16", sockets=2, callers_per_socket=16,
                     callers=32, workers=256, repeat=i, **dict(self.row, lat_p99_ms=str(i)))
                for d in ("powerlaw", "uniform") for i in (1, 3)]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            RUNNER.write_summary(out, rows)
            with (out / "summary.csv").open(newline="") as f:
                summary = list(csv.DictReader(f))
            self.assertEqual(len(summary), 2)
            self.assertTrue(all(r["runs"] == "2" and float(r["lat_p99_ms"]) == 2 for r in summary))

    def test_latest_report_uses_full_run_pointer(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                root = Path("build-host-layer/results")
                root.mkdir(parents=True)
                selected = (root / "scaled-test").resolve()
                (root / "latest-scaled.txt").write_text(str(selected) + "\n")
                with patch("sys.argv", ["scaled", "--report", "latest"]), patch.object(RUNNER, "report") as report:
                    RUNNER.main()
                report.assert_called_once_with(selected)
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
