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
if [[ -e "$root/.git" ]]; then
    git -C "$root" submodule update --init --depth 1 \
        third_party/json third_party/argparse third_party/pybind11 \
        third_party/cuembed third_party/dlpack
fi
cmake -S "$root" -B "$root/build-host-layer" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DNVE_DRIVERLESS_BUILD=ON \
    -DNVE_DISABLE_MPI=ON -DNVE_DISABLE_TORCH_BINDINGS=ON \
    -DNVE_DISABLE_PLUGINS=ON -DNVE_DISABLE_TESTS_AND_SAMPLES=ON \
    -DNVE_BUILD_HOST_BENCHMARK=ON -DNVE_SM_VERSION=80 -DFAC_NUM_TUS=32 "$@"
cmake --build "$root/build-host-layer" --target "${BUILD_TARGET:-host_layer_bench}" \
    --parallel "${BUILD_JOBS:-4}"
