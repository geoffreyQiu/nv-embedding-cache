#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runner tests without CUDA, a benchmark build, or a specific host topology."""
import csv
import importlib.util
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


if __name__ == "__main__":
    unittest.main()
