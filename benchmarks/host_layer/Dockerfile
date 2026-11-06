FROM nvcr.io/nvidia/pytorch:26.03-py3
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git numactl libnuma-dev libhwloc-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/nv-embedding-cache
ARG BUILD_JOBS=4
COPY . .
RUN BUILD_JOBS=${BUILD_JOBS} BUILD_TARGET=nve-common bash benchmarks/host_layer/build.sh
ARG NVE_BENCH_SOURCE_REV=unknown
RUN BUILD_JOBS=${BUILD_JOBS} bash benchmarks/host_layer/build.sh \
    -DNVE_BENCH_SOURCE_REV=${NVE_BENCH_SOURCE_REV} && mkdir -m 1777 /results
ENV NVIDIA_VISIBLE_DEVICES=void CUDA_VISIBLE_DEVICES=-1
ENTRYPOINT ["/workspace/nv-embedding-cache/build-host-layer/bin/host_layer_bench"]
CMD ["--help"]
