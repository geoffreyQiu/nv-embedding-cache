#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runner tests without CUDA, a benchmark build, or a specific host topology."""
import csv
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "host_layer_multicore", ROOT / "benchmarks/host_layer/multicore.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class TopologyTests(unittest.TestCase):
    # Non-contiguous sibling IDs; repeated core IDs in different sockets;
    # multiple memory nodes in each socket.
    rows = [(0, 0, 0, 0), (8, 0, 0, 0), (2, 1, 0, 1), (10, 1, 0, 1),
            (4, 0, 1, 2), (12, 0, 1, 2), (6, 1, 1, 3), (14, 1, 1, 3)]

    def cases(self, allowed=None, memory=None, smt=True, mode="inline"):
        return RUNNER.make_cases(self.rows,
            {r[0] for r in self.rows} if allowed is None else allowed,
            {0, 1, 2, 3} if memory is None else memory, smt, mode)

    def test_physical_core_and_smt_masks(self):
        cases = self.cases()
        self.assertEqual(cases["socket0-cores"]["cpus"], [0, 2])
        self.assertEqual(cases["server-cores"]["cpus"], [0, 2, 4, 6])
        self.assertEqual(cases["server-smt"]["callers"], 8)
        self.assertEqual(cases["single-core"]["memory_policy"], "membind")
        self.assertEqual(cases["socket0-cores"]["memory_policy"], "interleave")
        self.assertEqual(cases["socket0-cores"]["nodes"], [0, 1])

    def test_restricted_allocation_is_not_a_full_server(self):
        with self.assertRaisesRegex(ValueError, "physical cores"):
            self.cases(allowed={0, 8, 2, 10, 6, 14}, smt=False)
        with self.assertRaisesRegex(ValueError, "logical CPUs"):
            self.cases(allowed={0, 2, 4, 6})
        self.assertEqual(len(self.cases(allowed={0, 2, 4, 6}, smt=False)), 3)

    def test_memoryless_or_disallowed_node_requires_review(self):
        with self.assertRaisesRegex(ValueError, "local memory"):
            self.cases(memory={0, 1, 2})

    def test_pool_uses_one_shared_pool(self):
        cases = self.cases(smt=False, mode="pool")
        self.assertNotIn("single-core", cases)
        self.assertEqual(cases["server-cores"]["callers"], 4)
        self.assertEqual(cases["server-cores"]["workers"], 4)


class PreflightTests(unittest.TestCase):
    def run_probe(self, emitted_revision, verification="passed", returncode=0):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)

            def fake_run(cmd, **kwargs):
                self.assertIn("--disable_threadpool", cmd)
                self.assertEqual(cmd[cmd.index("--num_steps") + 1], "1")
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "-1")
                with Path(cmd[-1]).open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["nve_revision", "verification"])
                    writer.writerow([emitted_revision, verification])
                return subprocess.CompletedProcess(cmd, returncode)

            with patch.object(RUNNER.subprocess, "run", side_effect=fake_run):
                RUNNER.verify_binary(Path("benchmark"), "current", out)

    def test_matching_build(self):
        self.run_probe("current")

    def test_stale_build(self):
        with self.assertRaisesRegex(SystemExit, "Rebuild"):
            self.run_probe("old")

    def test_failed_verification(self):
        with self.assertRaisesRegex(SystemExit, "Invalid benchmark preflight"):
            self.run_probe("current", verification="failed")

    def test_failed_process(self):
        with self.assertRaisesRegex(SystemExit, "preflight failed"):
            self.run_probe("current", returncode=1)


class SocketTuningTests(unittest.TestCase):
    @staticmethod
    def topology(cores):
        return [(socket * cores + core + sibling * cores * 2, core, socket, socket)
                for socket in range(2) for core in range(cores) for sibling in range(2)]

    def cases(self, cores, **kwargs):
        rows = self.topology(cores)
        return RUNNER.make_socket_tuning_cases(rows, {r[0] for r in rows}, {0, 1}, **kwargs)

    def test_both_turin_sizes_use_full_pool_and_one_thread_per_core(self):
        for cores in (128, 192):
            with self.subTest(cores=cores):
                cases = self.cases(cores)
                self.assertEqual(len(cases), 4)
                for callers in (1, 2, 4):
                    case = cases[f"socket0-c{callers}-full-pool"]
                    self.assertEqual((case["mode"], case["callers"], case["workers"]),
                                     ("pool", callers, cores))
                    self.assertEqual(case["cpus"], list(range(cores)))
                    self.assertEqual((case["nodes"], case["memory_policy"]), ([0], "membind"))
                inline = cases["socket0-c64-inline"]
                self.assertEqual((inline["mode"], inline["callers"], inline["workers"]),
                                 ("inline", 64, 0))
                self.assertEqual(len(set(inline["cpus"])), 64)
                self.assertEqual(inline["cpus"][0], 0)
                self.assertEqual(inline["cpus"][-1], cores - 1)
                self.assertTrue(set(inline["cpus"]) <= set(range(cores)))

    def test_allowed_smt_siblings_can_represent_physical_cores(self):
        rows = self.topology(128)
        cases = RUNNER.make_socket_tuning_cases(rows, set(range(256, 512)), {0, 1})
        self.assertEqual(cases["socket0-c1-full-pool"]["cpus"], list(range(256, 384)))
        self.assertEqual(len(cases["socket0-c64-inline"]["cpus"]), 64)

    def test_small_socket_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 64"):
            self.cases(32)

    def test_custom_pool_only_cases_on_both_turins(self):
        for cores in (128, 192):
            with self.subTest(cores=cores):
                cases = self.cases(cores, pool_callers=(16, 32, 64), include_inline=False)
                self.assertEqual(list(cases), [f"socket0-c{c}-full-pool" for c in (16, 32, 64)])
                for case, callers in zip(cases.values(), (16, 32, 64)):
                    self.assertEqual((case["mode"], case["callers"], case["workers"]),
                                     ("pool", callers, cores))
                    self.assertEqual(case["cpus"], list(range(cores)))
                    self.assertEqual((case["nodes"], case["memory_policy"]), ([0], "membind"))

    def test_invalid_pool_caller_counts_are_rejected(self):
        for counts in ((), (0, 16), (-1,), (16, 16)):
            with self.subTest(counts=counts), self.assertRaisesRegex(ValueError, "positive and unique"):
                self.cases(128, pool_callers=counts, include_inline=False)

    def test_pool_options_require_socket_profile_and_valid_counts(self):
        for argv in (["--pool-only"], ["--pool-callers", "16"],
                     ["--profile", "socket-tuning", "--pool-callers", "0"],
                     ["--profile", "socket-tuning", "--pool-callers", "16", "16"]):
            with self.subTest(argv=argv), patch("sys.argv", ["runner", *argv]), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                RUNNER.main()
            self.assertEqual(error.exception.code, 2)

    def test_profile_rejects_smt(self):
        with patch("sys.argv", ["runner", "--profile", "socket-tuning", "--smt"]), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            RUNNER.main()
        self.assertEqual(error.exception.code, 2)

    def test_mixed_profile_runs_and_summarizes_all_four_cases(self):
        self.run_profile((1, 2, 4), True, [])

    def test_custom_pool_only_runs_and_summarizes_exactly_three_cases(self):
        self.run_profile((16, 32, 64), False,
                         ["--pool-callers", "16", "32", "64", "--pool-only"])

    def run_profile(self, pool_callers, include_inline, extra_args):
        cases = self.cases(128, pool_callers=pool_callers, include_inline=include_inline)
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            inline = "--disable_threadpool" in cmd
            callers = int(cmd[cmd.index("--threads") + 1])
            if inline:
                self.assertEqual(callers, 64)
                self.assertNotIn("--threadpool_workers", cmd)
                workers = 1  # NVE InlineThreadPool reports one worker.
            else:
                self.assertIn(callers, pool_callers)
                workers = int(cmd[cmd.index("--threadpool_workers") + 1])
                self.assertEqual(workers, 128)
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "-1")
            row = dict(nve_revision="current", verification="passed", threads=callers,
                       disable_threadpool=int(inline), threadpool_workers_effective=workers,
                       **{field: 1.25 for field in RUNNER.FIELDS})
            path = Path(cmd[cmd.index("--csv_filename") + 1])
            exists = path.exists()
            with path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
            return subprocess.CompletedProcess(cmd, 0)

        def fake_command(cmd):
            if cmd[0] == "git":
                return "current"
            if cmd[:2] == ["lscpu", "--online"]:
                return "\n".join(",".join(map(str, row)) for row in self.topology(128))
            return "fixture"

        def fake_read(path):
            if str(path) == "/proc/self/status":
                return "Mems_allowed_list:\t0-1"
            if str(path).endswith("/meminfo"):
                return "MemTotal: 1000000 kB"
            return "unavailable"

        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                binary = Path("build-host-layer/bin/host_layer_bench")
                binary.parent.mkdir(parents=True)
                binary.touch()
                with patch("sys.argv", ["runner", "--profile", "socket-tuning", *extra_args]), \
                     patch.object(RUNNER, "command", side_effect=fake_command), \
                     patch.object(RUNNER, "read", side_effect=fake_read), \
                     patch.object(RUNNER, "make_socket_tuning_cases", return_value=cases) as case_builder, \
                     patch.object(RUNNER, "verify_binary") as preflight, \
                     patch.object(RUNNER.subprocess, "run", side_effect=fake_run), \
                     contextlib.redirect_stdout(io.StringIO()) as output:
                    RUNNER.main()
                preflight.assert_called_once()
                self.assertEqual(tuple(case_builder.call_args.kwargs["pool_callers"]), pool_callers)
                self.assertEqual(case_builder.call_args.kwargs["include_inline"], include_inline)
                expected_modes = ["pool"] * len(pool_callers) + (["inline"] if include_inline else [])
                self.assertEqual(len(calls), 5 * len(expected_modes))
                summary = next(Path("build-host-layer/results").glob("*/summary.csv"))
                with summary.open(newline="") as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual([r["mode"] for r in rows], expected_modes)
                self.assertEqual([r["runs"] for r in rows], ["5"] * len(expected_modes))
                self.assertTrue((summary.parent / "results-for-chat.txt").is_file())
                self.assertNotIn("Later round reminder", output.getvalue())
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
