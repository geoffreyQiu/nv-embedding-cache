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
opts=()
[[ -z "${DOCKER_RUNTIME:-}" ]] || opts+=(--runtime "$DOCKER_RUNTIME")
[[ -z "${DOCKER_CPUS:-}" ]] || opts+=(--cpus "$DOCKER_CPUS")
[[ -z "${DOCKER_CPUSET:-}" ]] || opts+=(--cpuset-cpus "$DOCKER_CPUSET")
[[ -z "${DOCKER_MEMSET:-}" ]] || opts+=(--cpuset-mems "$DOCKER_MEMSET")
# Docker cp works with remote daemons and scratch directories absent on the daemon.
container=$(docker create --network none --user "$(id -u):$(id -g)" \
    --env NVIDIA_VISIBLE_DEVICES=void --env CUDA_VISIBLE_DEVICES=-1 "${opts[@]}" \
    "${HOSTLAYER_IMAGE:-nve-host-layer-bench:26.03}" "$@")
trap 'docker rm -f "$container" >/dev/null' EXIT
# Seed existing CSVs for sequential appends. Do not run this wrapper concurrently
# against the same results directory/CSV file.
docker cp "$root/build-host-layer/results/." "$container:/results/"
status=0
docker start -a "$container" || status=$?
# docker start may return zero even when the process exits with an error.
exit_code=$(docker inspect --format '{{.State.ExitCode}}' "$container")
docker cp "$container:/results/." "$root/build-host-layer/results/"
[[ "$status" -eq 0 ]] || exit "$status"
exit "$exit_code"
