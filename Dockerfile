# TANUH Buffer TEE — single static Go binary, distroless image.
#
# buffer-tee owns everything in-process: browser-facing RA-TLS HTTPS server
# (:8443, HPKE submit + AES-GCM chunked uploads), the persistent job store,
# the GPU-first/CPU-fallback provisioning state machine (Compute API with
# Operation polling), RA-TLS dispatch to the Processing TEE, the
# attestation-authenticated completion callback, and bounded crash recovery.
# The TLS serving cert is fetched from Secret Manager at startup (in-memory).
FROM golang:1.26-bookworm AS go-builder
WORKDIR /src
COPY Tanuh-buffer-tee-ratls/enclave/go.mod Tanuh-buffer-tee-ratls/enclave/go.sum ./
RUN go mod download
COPY Tanuh-buffer-tee-ratls/enclave/ ./
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" \
    -o /out/buffer-tee ./cmd/buffer-tee/

# distroless/static: CA certificates included (needed for Google APIs +
# Keycloak JWKS), no shell, no libc — the attested image is the binary.
FROM gcr.io/distroless/static-debian12

ENV BASE_DIR=/app \
    RATLS_SERVER_AUDIENCE=ratls-browser \
    RATLS_AUDIENCE=ratls-buffer-tee \
    GPU_CS_ADDR=10.128.0.37:443 \
    CPU_CS_INSTANCE=cpu-cs-tdx \
    MAX_GPU_PROVISION_ATTEMPTS=1 \
    SCHEDULER_INTERVAL_SECONDS=15

# Soft memory target for the Go runtime: uploads and dispatches stream through
# small buffers, so the process needs a few tens of MB whatever the model size.
ENV GOMEMLIMIT=1GiB

COPY --from=go-builder /out/buffer-tee /buffer-tee

LABEL "tee.launch_policy.allow_env_override"="RATLS_SERVER_AUDIENCE,GPU_CS_ADDR,GPU_CS_IMAGE_DIGEST,GPU_CS_INSTANCE,GPU_CS_ZONE,KEYCLOAK_JWKS_URL,KEYCLOAK_ISSUER,CPU_CS_ADDR,CPU_CS_IMAGE_DIGEST,CPU_CS_INSTANCE,CPU_CS_ZONE,MAX_GPU_PROVISION_ATTEMPTS,PROCESSING_VM_BOOT_TIMEOUT_SECONDS,CPU_VM_BOOT_TIMEOUT_SECONDS,DISPATCH_TIMEOUT_SECONDS,MAX_REQUEUE,BUFFER_CALLBACK_BASE,CALLBACK_AUDIENCE,PROJECT"
LABEL "tee.launch_policy.allow_cmd_override"="false"

EXPOSE 8443

ENTRYPOINT ["/buffer-tee"]
