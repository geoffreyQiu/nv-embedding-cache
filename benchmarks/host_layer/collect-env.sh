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
image=${HOSTLAYER_IMAGE:-nve-host-layer-bench:26.03}
{
    date -u --iso-8601=seconds
    docker image inspect "$image" --format 'Benchmark image: {{.Id}}'
    docker image inspect nvcr.io/nvidia/pytorch:26.03-py3 --format 'Base digests: {{json .RepoDigests}}'
    docker run --rm ${DOCKER_RUNTIME:+--runtime "$DOCKER_RUNTIME"} --network none \
        -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES=-1 --entrypoint bash "$image" -c '
            set -euo pipefail
            uname -a
            lscpu
            numactl --hardware
            cat /proc/self/status | sed -n "/Cpus_allowed_list/p; /Mems_allowed_list/p"
            cat /sys/fs/cgroup/cpu.max 2>/dev/null || true
            cat /sys/fs/cgroup/memory.max 2>/dev/null || true
            c++ --version
            nvcc --version
            cmake --version
            ldd build-host-layer/bin/host_layer_bench
            echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
            # Managed runtimes can inject /dev nodes even with NVIDIA visibility void.
            ls /dev/nvidia* 2>/dev/null || true
            python3 -c "import ctypes; r=ctypes.CDLL(\"libcudart.so.13\"); n=ctypes.c_int(); status=r.cudaGetDeviceCount(ctypes.byref(n)); print(\"CUDA device count:\", n.value, \"status:\", status); assert n.value == 0"
            sha256sum benchmarks/host_layer_bench.cpp build-host-layer/bin/host_layer_bench
        '
} > "$root/build-host-layer/results/environment.txt"
cat "$root/build-host-layer/results/environment.txt"
