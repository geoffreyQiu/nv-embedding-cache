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
git -C "$root" submodule update --init --depth 1 \
    third_party/json third_party/argparse third_party/pybind11 \
    third_party/cuembed third_party/dlpack
revision=$(git -C "$root" rev-parse HEAD)
if [[ -n "$(git -C "$root" status --porcelain --untracked-files=normal)" ]]; then
    revision="$revision-dirty"
fi
docker build --progress=plain --build-arg "BUILD_JOBS=${BUILD_JOBS:-4}" \
    --build-arg "NVE_BENCH_SOURCE_REV=$revision" \
    -t "${HOSTLAYER_IMAGE:-nve-host-layer-bench:26.03}" \
    -f "$root/benchmarks/host_layer/Dockerfile" "$root" "$@"
