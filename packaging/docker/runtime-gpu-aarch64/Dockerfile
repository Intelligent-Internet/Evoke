FROM ubuntu:24.04 AS build

RUN set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        libicu-dev \
        libjansson-dev \
        libssl-dev \
        make \
        pkg-config; \
    rm -rf /var/lib/apt/lists/*

COPY --from=ii42_gpu_rootfs . /
COPY --from=evoke_onnxruntime_headers include/ /usr/local/include/

WORKDIR /src

COPY Makefile CMakeLists.txt ii42.control LICENSE README.md ./
COPY src ./src
COPY packaging/onnxruntime.version ./packaging/onnxruntime.version

RUN PKG_CONFIG_PATH=/usr/local/lib/pkgconfig \
        make runtime-server PG_CONFIG=/no/such/pg_config

FROM ubuntu:24.04

RUN set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        libicu74 \
        libjansson4 \
        libssl3 \
        libstdc++6; \
    rm -rf /var/lib/apt/lists/*

COPY --from=ii42_gpu_rootfs . /
COPY --from=build /src/build/ii42-runtime-server \
    /usr/local/bin/ii42-runtime-server
COPY --from=evoke_milestone_model . \
    /usr/local/share/ii42/models/default/
COPY packaging/docker/runtime-gpu-aarch64/evoke-gpu-runtime-entrypoint \
    /usr/local/bin/evoke-gpu-runtime-entrypoint

RUN set -eux; \
    chmod 0755 /usr/local/bin/ii42-runtime-server; \
    chmod 0755 /usr/local/bin/evoke-gpu-runtime-entrypoint; \
    printf '%s\n' \
        /usr/local/lib \
        /usr/local/cuda/targets/sbsa-linux/lib \
        > /etc/ld.so.conf.d/ii42-gpu.conf; \
    ldconfig

ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/targets/sbsa-linux/lib:/usr/lib/aarch64-linux-gnu

EXPOSE 8042

ENTRYPOINT ["/usr/local/bin/evoke-gpu-runtime-entrypoint"]
