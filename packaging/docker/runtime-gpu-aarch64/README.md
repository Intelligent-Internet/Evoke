# Evoke aarch64 GPU direct runtime image

This image is the Spark accelerator shape: it runs only the direct
`ii42-runtime-server` process. It does not start PostgreSQL, does not install
the ii42 extension SQL, and does not own relation storage. The coordinator
database remains the only authority for indexes and model checkout identity.

The image compiles the server binary from this source tree in a build stage,
then copies only:

- `/usr/local/bin/ii42-runtime-server`;
- the default model checkout named build context into
  `/usr/local/share/ii42/models/default`;
- a Linux aarch64 CUDA 13 / SM121 ONNX Runtime 1.29 C artifact and CUDA runtime
  libraries from the `ii42_gpu_rootfs` named build context;
- the matching ONNX Runtime headers from the `ii42_onnxruntime_headers` named
  build context;
- `ii42-gpu-runtime-entrypoint`, which starts the direct HTTP service.

The `ii42_gpu_rootfs` build context must provide GPU ONNX Runtime and CUDA
libraries using the target filesystem layout. Microsoft does not publish a
Linux aarch64 GPU archive for ONNX Runtime 1.29, so this input must be built
from the pinned upstream tag and independently checksum-locked before use.
Do not reuse the former 1.26 SM121 artifact. Install the qualified custom
archive into a staging root with a manifest that contains its exact checksum:

```bash
scripts/install_onnxruntime_c.sh \
    --version 1.29.0 \
    --archive /path/to/onnxruntime-linux-aarch64-gpu-cuda13-sm121-1.29.0.tar.gz \
    --prefix /tmp/ii42-gpu-rootfs/usr/local
```

Build the image from the repository root and pass the runtime root, matching
ONNX Runtime headers, and exact milestone checkout as named contexts:

```bash
docker build \
    -f packaging/docker/runtime-gpu-aarch64/Dockerfile \
    --build-context ii42_gpu_rootfs=/tmp/ii42-gpu-rootfs \
    --build-context ii42_onnxruntime_headers=/path/to/onnxruntime-1.29.0 \
    --build-context ii42_milestone_model=/path/to/evoke-milestone-model \
    -t evoke-runtime-gpu:direct-cuda13 \
    .
```

The `ii42_onnxruntime_headers` context must expose
`include/onnxruntime/onnxruntime_c_api.h` for the same ONNX Runtime build as
the runtime libraries in `ii42_gpu_rootfs`.

Run the image with NVIDIA container runtime:

```bash
docker run -d --name evoke-runtime-gpu --gpus all --ipc=host \
    -p 18042:8042 \
    -e II42_CHECKOUT_SIGNATURE="$CHECKOUT_SIGNATURE" \
    -e II42_HTTP_WORKER_COUNT=8 \
    -e II42_HTTP_QUEUE_DEPTH=48 \
    -e II42_MAX_BATCH_SIZE=128 \
    -e II42_MAX_DELAY_MS=5 \
    -e II42_MAX_REQUEST_BYTES=67108864 \
    evoke-runtime-gpu:direct-cuda13
```

`II42_CHECKOUT_SIGNATURE` is required. The direct service intentionally has no
PostgreSQL dependency, so it cannot reconstruct PostgreSQL `jsonb` checkout
identity by itself. Pass the signature produced by the coordinator database
for the exact model checkout.

`II42_MODEL_PATH` defaults to `/usr/local/share/ii42/models/default`.

`II42_MAX_BATCH_SIZE` defaults to 128 and can be raised up to 512. It is a
service throughput limit, not a model checkout field, so different accelerator
hosts may choose different batching without changing checkout signatures or
index identities. Larger batches should be benchmarked per host; for this
model they are not guaranteed to improve throughput.

`II42_HTTP_QUEUE_DEPTH` controls runtime-server backpressure. When omitted,
the server derives a bounded queue from `II42_HTTP_WORKER_COUNT`; when the
queue is full, `/v1/encode` returns HTTP 503 so callers can retry another
accelerator or reduce in-flight work. This is a server-capacity setting, not a
model identity field.

`II42_INTRA_OP_THREADS` is optional. Leave it unset for the default ONNX
Runtime policy on GPU hosts unless profiling shows CPU-side tokenization or
fallback nodes need a tighter cap.

`/health` and `/v1/models` report `runtime_backend: "direct"`. After startup,
Spark GPU hosts should report `active_provider: "cuda"` or `"tensorrt"`.
