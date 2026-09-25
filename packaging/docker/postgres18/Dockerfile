FROM postgres:18 AS build

ARG ONNXRUNTIME_VERSION

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        curl \
        libicu-dev \
        libjansson-dev \
        make \
        pkg-config \
        python3 \
        postgresql-server-dev-18 \
    && rm -rf /var/lib/apt/lists/*

COPY scripts/install_onnxruntime_c.sh /usr/local/bin/install-onnxruntime-c
COPY packaging/onnxruntime.sha256 \
    /usr/local/share/ii42/onnxruntime.sha256

RUN test -n "${ONNXRUNTIME_VERSION}" \
    && /usr/local/bin/install-onnxruntime-c \
        --version "${ONNXRUNTIME_VERSION}" \
        --prefix /usr/local

WORKDIR /src

COPY Makefile CMakeLists.txt ii42.control LICENSE README.md ./
COPY src ./src
COPY sql ./sql
COPY packaging/onnxruntime.version ./packaging/onnxruntime.version
COPY packaging/milestone-model.json ./packaging/milestone-model.json
COPY packaging/MILESTONE-MODEL-NOTICE \
    ./packaging/MILESTONE-MODEL-NOTICE
COPY scripts/validate_milestone_model_checkout.py \
    ./scripts/validate_milestone_model_checkout.py
COPY --from=evoke_milestone_model . /tmp/evoke-milestone-model

RUN make clean PG_CONFIG=/usr/bin/pg_config \
    && PKG_CONFIG_PATH=/usr/local/lib/pkgconfig \
        make II42_ENABLE_ONNXRUNTIME=1 PG_CONFIG=/usr/bin/pg_config \
    && make runtime-server PG_CONFIG=/usr/bin/pg_config \
    && PKG_CONFIG_PATH=/usr/local/lib/pkgconfig \
        make install II42_ENABLE_ONNXRUNTIME=1 \
        DESTDIR=/tmp/ii42-stage \
        PG_CONFIG=/usr/bin/pg_config \
    && python3 scripts/validate_milestone_model_checkout.py \
        --checkout /tmp/evoke-milestone-model \
        --quiet \
    && model_dir="$(pg_config --sharedir)/ii42/models/default" \
    && mkdir -p "/tmp/ii42-stage${model_dir}" \
        /tmp/ii42-stage/usr/local/share/licenses/ii42-model \
    && cp -a /tmp/evoke-milestone-model/. \
        "/tmp/ii42-stage${model_dir}/" \
    && cp LICENSE \
        /tmp/ii42-stage/usr/local/share/licenses/ii42-model/LICENSE \
    && cp packaging/MILESTONE-MODEL-NOTICE \
        /tmp/ii42-stage/usr/local/share/licenses/ii42-model/NOTICE \
    && install -D -m 0755 build/ii42-runtime-server \
        /tmp/ii42-stage/usr/local/bin/ii42-runtime-server

FROM postgres:18

ARG II42_VERSION
ARG ONNXRUNTIME_VERSION
ARG II42_GIT_COMMIT
ARG II42_GIT_TREE_STATE
ARG II42_MODEL_ID
ARG II42_MODEL_MANIFEST_SHA256

LABEL org.opencontainers.image.title='Evoke PostgreSQL 18'
LABEL org.opencontainers.image.description='PostgreSQL 18 image with Evoke preinstalled'
LABEL org.opencontainers.image.version="${II42_VERSION}"
LABEL org.opencontainers.image.source='https://github.com/Intelligent-Internet/Evoke'
LABEL org.opencontainers.image.revision="${II42_GIT_COMMIT}"
LABEL org.opencontainers.image.onnxruntime.version="${ONNXRUNTIME_VERSION}"
LABEL io.evoke.git-tree-state="${II42_GIT_TREE_STATE}"
LABEL io.evoke.milestone-model.id="${II42_MODEL_ID}"
LABEL io.evoke.milestone-model.manifest-sha256="${II42_MODEL_MANIFEST_SHA256}"

COPY --from=build /tmp/ii42-stage/ /
COPY --from=build /usr/local/lib/ /usr/local/lib/
COPY --from=build /usr/local/share/licenses/onnxruntime/ \
    /usr/local/share/licenses/onnxruntime/
COPY packaging/docker/postgres18/initdb/010-create-ii42.sh \
    /docker-entrypoint-initdb.d/010-create-ii42.sh

RUN apt-get update \
    && apt-get install -y --no-install-recommends libjansson4 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/postgresql \
    && printf '%s\n' '/usr/local/lib' \
        > /etc/ld.so.conf.d/onnxruntime.conf \
    && ldconfig \
    && chown postgres:postgres /var/run/postgresql \
    && chmod 3775 /var/run/postgresql \
    && chmod 0755 /docker-entrypoint-initdb.d/010-create-ii42.sh
