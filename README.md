# TANUH Buffer TEE

Always-on Confidential Space VM that sits between the browser and the Processing TEE. It handles attestation, encrypted job intake, file staging, queueing, provisioning, and secure dispatch — the browser never communicates with the Processing TEE directly.

The runtime is a **single static Go binary** (`buffer-tee`) in a distroless image. There is no Python, no shell, no sidecar processes.

## Architecture

```
Browser
  │  HPKE-encrypted metadata   (POST /v1/submit)
  │  AES-256-GCM chunked files (PUT  /v1/upload/:job_id/model|weights|adaptor)
  ▼
buffer-tee (Go, :8443, TLS 1.3, Keycloak JWT)
  ├─ HPKE / AES-GCM decrypt + SHA-256 verification
  ├─ job store + ordered queue   (cvm_workflow/buffer/, JSON on disk)
  ├─ scheduler (every 15 s): retention cleanup, stall warnings,
  │    bounded crash recovery, dispatch cycle
  ├─ provisioning: GPU-first (capped stop→start attempts, Compute API with
  │    Operation polling → quota errors fail over fast) → CPU fallback
  ├─ RA-TLS dispatch (in-process): verify Processing TEE attestation
  │    (image digest, hwmodel, EKM channel binding) → POST /api/load-model
  └─ completion callback receiver  (POST /v1/jobs/:id/complete,
       authenticated by the Processing TEE's CS attestation token)
```

## Job lifecycle

1. Browser POSTs HPKE-encrypted `{dataset_id, model_format, model_sha256, adaptor_sha256[, weights_sha256][, input_spec_sha256]}` → job `pending_upload`.
   `model_format` is `onnx`, `torchscript` or `huggingface`; `weights_sha256` (the `.onnx.data` file) is ONNX-only and optional;
   `input_spec_sha256` (an `input_spec.json` giving the input size / resize, e.g. `{"input_size": [224, 224], "resize": "bicubic"}`)
   is optional for ONNX and TorchScript. A TorchScript model needs it unless the size is embedded in the `.pt`.
2. Encrypted chunked uploads, one per slot (see `internal/jobs/artifacts.go`); each must match its submit-time SHA-256 — a slot with no commitment accepts nothing → `queued` once every expected slot is present

   | Slot | onnx | torchscript | huggingface |
   |---|---|---|---|
   | `model` | `.onnx` (required) | `.pt` (required) | `.zip` (required) |
   | `weights` | `.onnx.data` (if committed) | — | — |
   | `adaptor` | `adaptor.py` (required) | `adaptor.py` (required) | `adaptor.py` (required) |
   | `input_spec` | `input_spec.json` (if committed) | `input_spec.json` (if committed) | — |
3. Scheduler re-checks every stored artifact against its commitment (a mismatch fails the job), provisions a Processing TEE (GPU attempts → CPU fallback) and dispatches payload v2 over RA-TLS → `dispatched`
4. Processing TEE runs the eval, submits results to the leaderboard, then POSTs its terminal status here → `complete` or `error` (with the classified failure)
5. If the TEE dies without calling back: after `DISPATCH_TIMEOUT_SECONDS` with the TEE unhealthy the job is requeued, at most `MAX_REQUEUE` times, then marked `error` — nothing can loop forever.

## API (:8443)

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/v1/attest` | Keycloak JWT | RA-TLS attestation bundle (OIDC token + HPKE pub + liveness sig) |
| `POST` | `/v1/submit` | Keycloak JWT | HPKE-encrypted job metadata → `{job_id}` |
| `PUT` | `/v1/upload/:job_id/:slot` | Keycloak JWT | AES-256-GCM chunked upload; `slot` = `model` \| `weights` \| `adaptor` \| `input_spec` (anything else, or a slot the model format doesn't use, is 404) |
| `GET` | `/v1/status/:job_id` | Keycloak JWT | Job record |
| `GET` | `/v1/queue` | Keycloak JWT + `org_admin` | All jobs + queue |
| `GET` | `/v1/results/:job_id` | Keycloak JWT | Job status (results live on the leaderboard) |
| `POST` | `/v1/jobs/:job_id/complete` | **CS attestation token** | Terminal status from the Processing TEE. Token must verify against Google's CS JWKS, carry `image_digest` ∈ {GPU, CPU expected digests}, and its `eat_nonce` must equal `hex(sha256(body))` — bound to the exact payload. |
| `GET` | `/healthz` | — | Health check |

## Environment variables

Passed via Confidential Space metadata (`tee-env-` prefix). The image bakes only safe defaults; digests must always come from metadata.

| Variable | Default | Description |
|---|---|---|
| `GPU_CS_ADDR` / `CPU_CS_ADDR` | — | Processing TEE RA-TLS `ip:port` |
| `GPU_CS_IMAGE_DIGEST` / `CPU_CS_IMAGE_DIGEST` | — (required) | Expected attested image digests (dispatch pins them) |
| `GPU_CS_INSTANCE`/`GPU_CS_ZONE`, `CPU_CS_INSTANCE`/`CPU_CS_ZONE` | gpu-cs-tdx-h100 / cpu-cs-tdx, us-central1-a | Compute start/stop targets |
| `MAX_GPU_PROVISION_ATTEMPTS` | 1 | GPU stop→start attempts per job before CPU fallback |
| `PROCESSING_VM_BOOT_TIMEOUT_SECONDS` / `CPU_VM_BOOT_TIMEOUT_SECONDS` | 300 | Boot-health wait per attempt |
| `DISPATCH_TIMEOUT_SECONDS` | 600 | No-callback threshold before crash recovery |
| `MAX_REQUEUE` | 2 | Requeue budget per job, then `error` |
| `BUFFER_CALLBACK_BASE` | https://tee.dev.tanuh.iudx.io:8443 | Base URL the Processing TEE calls back to |
| `CALLBACK_AUDIENCE` | tanuh-buffer-callback | Required `aud` in the callback attestation token |
| `KEYCLOAK_JWKS_URL` / `KEYCLOAK_ISSUER` | — | Browser JWT validation (empty disables auth) |
| `TLS_CERT` / `TLS_KEY` | — | Local-dev cert files; unset → fetched from Secret Manager (`tanuh-tls-cert`/`tanuh-tls-key`) in memory |

## Build & deploy

```bash
IMAGE=us-central1-docker.pkg.dev/p3dx-depa-sandbox/ratls/buffer-tee
docker build -t $IMAGE:<tag> . && docker push $IMAGE:<tag>
docker inspect --format='{{index .RepoDigests 0}}' $IMAGE:<tag>

gcloud compute instances add-metadata buffer-tee-vm-tdx --zone=us-east1-c \
  --metadata tee-image-reference=$IMAGE@sha256:<digest>
```

Every buffer redeploy changes the digest the browser pins — update `EXPECTED_IMAGE_DIGEST` in `Tanuh-browser-ratls` `constants.ts` and rebuild the UI.

## Debug

All state persists under `/app/cvm_workflow/buffer/` (same layout as the former Python manager — a rollback reads the same files):
- `jobs/<job_id>/job.json` — job record
- `queue.json` — ordered queue
- `outgoing/<job_id>-secure-dispatch.json` — staged dispatch payload
- `runtime/dispatch_state.json` — last dispatch breadcrumbs

Serial log prefixes: `buffer-tee:` (server), `scheduler:`, `provision:`, `dispatch:`, `server:`.
