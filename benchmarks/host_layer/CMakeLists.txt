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

add_executable(host_layer_bench ../host_layer_bench.cpp)
target_include_directories(host_layer_bench PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/argparse/include)
target_link_libraries(host_layer_bench PRIVATE nve-common)

set(NVE_BENCH_SOURCE_REV "" CACHE STRING "Source revision recorded by host_layer_bench")
if(NOT NVE_BENCH_SOURCE_REV)
  set(NVE_BENCH_SOURCE_REV "unknown")
  if(EXISTS "${PROJECT_SOURCE_DIR}/.git")
    execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      OUTPUT_VARIABLE _host_bench_rev OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _host_bench_git_status ERROR_QUIET)
    if(_host_bench_git_status EQUAL 0)
      set(NVE_BENCH_SOURCE_REV "${_host_bench_rev}")
    endif()
  endif()
endif()
target_compile_definitions(host_layer_bench PRIVATE NVE_SOURCE_REV="${NVE_BENCH_SOURCE_REV}")
