#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Placement and workflow checks; real NVE lookup tests are in test_host_layer_bench.py."""
import contextlib
import csv
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('host_layer_run', ROOT/'benchmarks/host_layer/run.py')
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def topology(p):
    return [(s*p+c+hw*2*p, 805306368+c*131072, 268435456+s*16777216, s)
            for s in range(2) for c in range(p) for hw in range(2)]


class RunnerTests(unittest.TestCase):
    def test_three_products_both_scopes_no_smt_siblings(self):
        for p in (88, 128, 192):
            rows = topology(p)
            lookup = {r[0]:r for r in rows}
            for allowed in (set(range(4*p)), set(range(2*p, 4*p))):
                plan = RUNNER.make_cases(rows, allowed, {0, 1})
                self.assertEqual(set(plan), set(RUNNER.CASES))
                for name, case in plan.items():
                    sockets = 2 if name.startswith('server') else 1
                    c = p if name.endswith('full') else (64 if name.endswith('64') else 16)
                    self.assertEqual(len(case['callers']), c*sockets)
                    self.assertEqual(len(case['workers']), p*sockets if 'pool' in name else 0)
                    self.assertEqual(len({(lookup[cpu][2],lookup[cpu][1]) for cpu in case['cpus']}),len(case['cpus']))
                    for node in range(sockets):
                        self.assertEqual(sum(lookup[cpu][3]==node for cpu in case['callers']),c)
                    self.assertEqual(case['nodes'],list(range(sockets)))
                    self.assertEqual(case['policy'],'membind' if sockets==1 else 'interleave')

    def test_inaccessible_core_or_memory_rejected(self):
        rows = topology(88)
        for allowed, nodes in ((set(range(1,176)), {0,1}), (set(range(352)), {0})):
            with self.assertRaises(ValueError):
                RUNNER.make_cases(rows,allowed,nodes)

    def test_multiple_numa_nodes_per_socket(self):
        rows=[(cpu,core,socket,node*2+cpu%2) for cpu,core,socket,node in topology(128)]
        plan=RUNNER.make_cases(rows,set(range(512)),{0,1,2,3})
        self.assertEqual(plan['socket-inline-full']['nodes'],[0,1])
        self.assertEqual(plan['server-inline-full']['nodes'],[0,1,2,3])
        self.assertEqual(plan['socket-inline-full']['policy'],'interleave')

    def test_smoke_is_small_and_accessible(self):
        plan=RUNNER.make_cases(topology(88),{3,5,90,92},{0,1},smoke=True)
        self.assertEqual(plan['server-pool-16']['workers'],[3,5,90,92])
        self.assertEqual(plan['socket-inline-64']['callers'],[3,5])

    def test_commands_preserve_workload_counts_and_binding(self):
        plan=RUNNER.make_cases(topology(128),set(range(512)),{0,1})
        args=SimpleNamespace(rows=33554432,steps=12345,warmup=100)
        for case in plan.values():
            cmd=RUNNER.bench_command(Path('bench'),case,args,Path('out.csv'))
            for key,value in (('--alpha','1.15'),('--num_steps','12345'),('--latency_steps','12345'),
                              ('--key_buffer_steps','512'),('--num_table_rows','33554432')):
                self.assertEqual(cmd[cmd.index(key)+1],value)
            self.assertEqual(cmd[1],'--physcpubind='+','.join(map(str,case['cpus'])))
            self.assertEqual(cmd[2],'--'+case['policy']+'='+','.join(map(str,case['nodes'])))
            self.assertEqual('--disable_threadpool' in cmd,not case['workers'])
            self.assertNotIn('--throughput_seconds',cmd)

    def test_powerlaw_generator_stays_aligned_with_single_layer(self):
        def generator(path):
            source=path.read_text(); start=source.index('class PowerLaw final {')
            end=source.index('\n};',source.index('class KeyGenerator final {',start))+3
            return source[start:end].replace('const RunConfig& cfg','const Config& cfg')
        self.assertEqual(generator(ROOT/'benchmarks/host_layer_bench.cpp'),generator(ROOT/'benchmarks/single_layer_bench.cpp'))

    def test_summary_does_not_read_topology_or_its_own_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            (out/'topology.csv').write_text('CPU,CORE,SOCKET,NODE\n')
            row=dict(verification='passed',num_steps='100',num_table_rows='33554432',threads='64',
                     disable_threadpool='1',threadpool_workers_effective='1',
                     mkps='10',gbps='5.12',lat_mean_ms='2',inference_time_s='3',latency_time_s='4')
            with (out/'socket-inline-64.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
                writer.writerow(row);writer.writerow(dict(row,mkps='20'))
            for _ in range(2):
                with contextlib.redirect_stdout(io.StringIO()): RUNNER.report(out)
                with (out/'summary.csv').open() as f:
                    saved=list(csv.DictReader(f))
                self.assertEqual(len(saved),1)
                self.assertEqual(saved[0]['workers'],'0')
                self.assertEqual(saved[0]['runs'],'2')
                self.assertEqual(float(saved[0]['mkps']),15)

    def test_invalid_workload_options_fail_before_execution(self):
        for opts in (['--steps','0'],['--warmup','-1'],['--rows','0'],['--distributions','uniform']):
            with patch('sys.argv',['run',*opts]), patch.object(RUNNER,'command') as command, \
                 contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                RUNNER.main()
            command.assert_not_called()


if __name__=='__main__':
    unittest.main()
