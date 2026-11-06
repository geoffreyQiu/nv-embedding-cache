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
docker run --rm ${DOCKER_RUNTIME:+--runtime "$DOCKER_RUNTIME"} --network none \
    -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES=-1 --entrypoint python3 \
    "${HOSTLAYER_IMAGE:-nve-host-layer-bench:26.03}" \
    tests/benchmarks/test_host_layer_bench.py build-host-layer/bin/host_layer_bench \
    2>&1 | tee "$root/build-host-layer/results/smoke-test.log"
