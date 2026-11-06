#!/usr/bin/env bash
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

set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
mkdir -p "$root/build-host-layer/results"
# Change environment variables rather than editing the script. Run sequentially.
for alpha in ${ALPHAS:-0 0.8 1.05 1.2}; do
    for callers in ${CALLERS:-1 2 4 8}; do
        for rep in $(seq 1 "${REPEATS:-3}"); do
            bash "$root/benchmarks/host_layer/docker-run.sh" \
                --num_table_rows "${ROWS:-1000000}" --embedding_dim "${DIM:-128}" \
                --data_type "${DTYPE:-float32}" --batch "${BATCH:-256}" \
                --hotness "${HOTNESS:-1024}" --num_steps "${STEPS:-100}" \
                --num_warmup_steps "${WARMUP:-100}" --alpha "$alpha" --threads "$callers" \
                --disable_threadpool --tag "inline-rep-$rep" \
                --csv_filename /results/sweep.csv
        done
    done
done
