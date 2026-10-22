package server

import (
	"bufio"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"syscall"
	"time"

	"github.com/datakaveri/tanuh-buffer-tee/internal/bundle"
	"github.com/datakaveri/tanuh-buffer-tee/internal/chunked"
	hpkepkg "github.com/datakaveri/tanuh-buffer-tee/internal/hpke"
	"github.com/datakaveri/tanuh-buffer-tee/internal/jobs"
)

const aesKeyLen = 32

// JobRequest is the small HPKE-encrypted metadata payload.
// Model bytes are NOT included — they are uploaded separately via
// PUT /v1/upload/{job_id}/{slot}. SHA256 hashes commit to the files before
// they arrive (prevents in-transit swaps); an upload is accepted only if it
// matches its committed hash.
type JobRequest struct {
	DatasetID     string `json:"dataset_id"`               // catalogue dataset UUID (passed through)
	ModelFormat   string `json:"model_format"`             // onnx | torchscript | huggingface
	ModelSHA256   string `json:"model_sha256"`             // .onnx / TorchScript .pt / Hugging Face .zip
	WeightsSHA256 string `json:"weights_sha256,omitempty"` // ONNX external weights (.onnx.data), optional
	AdaptorSHA256 string `json:"adaptor_sha256"`           // adaptor.py
	// InputSpecSHA256 commits to an optional input_spec.json (ONNX /
	// TorchScript input size and resize method).
	InputSpecSHA256 string `json:"input_spec_sha256,omitempty"`
}

// validate checks the commitments against the model format so a job that can
// never be dispatched is refused at submit rather than failing later.
func (r *JobRequest) validate() error {
	if !jobs.ValidFormat(r.ModelFormat) {
		return fmt.Errorf("model_format must be one of onnx, torchscript, huggingface (got %q)", r.ModelFormat)
	}
	r.ModelSHA256 = strings.ToLower(strings.TrimSpace(r.ModelSHA256))
	r.WeightsSHA256 = strings.ToLower(strings.TrimSpace(r.WeightsSHA256))
	r.AdaptorSHA256 = strings.ToLower(strings.TrimSpace(r.AdaptorSHA256))
	if !isSHA256Hex(r.ModelSHA256) {
		return fmt.Errorf("model_sha256 must be a 64-character hex SHA-256")
	}
	if !isSHA256Hex(r.AdaptorSHA256) {
		return fmt.Errorf("adaptor_sha256 must be a 64-character hex SHA-256")
	}
	if r.WeightsSHA256 != "" {
		if r.ModelFormat != jobs.FormatONNX {
			return fmt.Errorf("weights_sha256 is only used with model_format onnx")
		}
		if !isSHA256Hex(r.WeightsSHA256) {
			return fmt.Errorf("weights_sha256 must be a 64-character hex SHA-256")
		}
	}
	r.InputSpecSHA256 = strings.ToLower(strings.TrimSpace(r.InputSpecSHA256))
	if r.InputSpecSHA256 != "" {
		if r.ModelFormat == jobs.FormatHuggingFace {
			return fmt.Errorf("input_spec_sha256 is only used with onnx and torchscript " +
				"(Hugging Face models use their preprocessor_config.json)")
		}
		if !isSHA256Hex(r.InputSpecSHA256) {
			return fmt.Errorf("input_spec_sha256 must be a 64-character hex SHA-256")
		}
	}
	return nil
}

func isSHA256Hex(s string) bool {
	if len(s) != 64 {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}

// extractKeycloakSub decodes the JWT payload (already validated by middleware)
// and returns the "sub" claim, or empty string on any error.
func extractKeycloakSub(tokenStr string) string {
	parts := strings.Split(tokenStr, ".")
	if len(parts) != 3 {
		return ""
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return ""
	}
	var claims struct {
		Sub string `json:"sub"`
	}
	if err := json.Unmarshal(payload, &claims); err != nil {
		return ""
	}
	return claims.Sub
}

// HandleSubmit serves POST /v1/submit.
// Receives HPKE-encrypted metadata only — no model bytes in this call.
// Returns {job_id, status} encrypted back to browser.
func (s *Server) HandleSubmit(w http.ResponseWriter, r *http.Request) {
	browserPubB64u := r.Header.Get("X-RATLS-Browser-HPKE")
	if browserPubB64u == "" {
		http.Error(w, "missing X-RATLS-Browser-HPKE header", http.StatusBadRequest)
		return
	}
	browserPubBytes, err := base64.RawURLEncoding.DecodeString(browserPubB64u)
	if err != nil || len(browserPubBytes) != 32 {
		http.Error(w, "bad X-RATLS-Browser-HPKE: must be 32-byte base64url", http.StatusBadRequest)
		return
	}

	// 1 MB limit — metadata only, no model bytes
	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20))
	if err != nil {
		http.Error(w, "failed to read body", http.StatusBadRequest)
		return
	}

	var req bundle.EncryptedRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}

	encBytes, err := base64.StdEncoding.DecodeString(req.Enc)
	if err != nil {
		http.Error(w, "bad enc", http.StatusBadRequest)
		return
	}
	ct, err := base64.StdEncoding.DecodeString(req.Ciphertext)
	if err != nil {
		http.Error(w, "bad ciphertext", http.StatusBadRequest)
		return
	}
	aad, err := base64.StdEncoding.DecodeString(req.AAD)
	if err != nil {
		http.Error(w, "bad aad", http.StatusBadRequest)
		return
	}

	hpkePriv, hpkePub := s.builder.HPKEKeys()
	info := hpkepkg.ContextInfo(hpkePub.Bytes(), browserPubBytes)

	plaintext, err := hpkepkg.OpenFromBrowser(hpkePriv, encBytes, info, aad, ct)
	if err != nil {
		http.Error(w, "decryption failed", http.StatusBadRequest)
		return
	}

	rawKeycloakToken := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
	keycloakSub := extractKeycloakSub(rawKeycloakToken)
	response := s.processJob(plaintext, keycloakSub, rawKeycloakToken)

	rid := r.Header.Get("X-Request-Id")
	respAAD := []byte(fmt.Sprintf(`{"rid":%q,"ts":%d}`, rid, time.Now().Unix()))

	sealed, err := hpkepkg.SealToBrowser(browserPubBytes, info, respAAD, response)
	if err != nil {
		http.Error(w, "encryption failed", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(bundle.EncryptedResponse{ //nolint:errcheck
		Enc:        base64.StdEncoding.EncodeToString(sealed.Enc),
		Ciphertext: base64.StdEncoding.EncodeToString(sealed.Ciphertext),
		AAD:        base64.StdEncoding.EncodeToString(respAAD),
	})
}

// HandleUpload serves PUT /v1/upload/{job_id}/{slot} for every upload slot
// (model, weights, adaptor, input_spec — see jobs.Artifacts). The
// AES-256-GCM-CHUNKED-v1 stream is written to disk exactly as uploaded, still
// encrypted, while its framing is checked (every chunk the header announces,
// nothing after); its AES key is unwrapped from HPKE and kept in memory by
// the store. Nothing is decrypted here: the Processing TEE decrypts the
// stream and verifies it against the submit-time SHA-256. Memory use is a
// small copy buffer whatever the file's size.
func (s *Server) HandleUpload(w http.ResponseWriter, r *http.Request) {
	jobID, slot := r.PathValue("job_id"), r.PathValue("slot")
	started := time.Now()
	dir, err := s.store.BeginUpload(jobID, slot)
	if err != nil {
		uploadError(w, err)
		return
	}
	defer s.store.EndUpload(jobID, slot)
	if r.ContentLength > 0 {
		if err := checkDiskSpace(dir, r.ContentLength); err != nil {
			writeJSON(w, http.StatusInsufficientStorage, map[string]any{"status": "error", "message": err.Error()})
			return
		}
	}

	rc := http.NewResponseController(w)
	body := &idleDeadlineReader{r: r.Body, rc: rc}
	up, err := s.receiveUpload(body, r.Header.Get("X-RATLS-Browser-HPKE"), dir, slot)
	// The upload may have outlasted the server's WriteTimeout; leave time to reply.
	_ = rc.SetWriteDeadline(time.Now().Add(time.Minute))
	if err != nil {
		log.Printf("upload %s/%s rejected: %v", jobID, slot, err)
		code := http.StatusBadRequest
		var we *writeError
		if errors.As(err, &we) {
			code = http.StatusInternalServerError
		}
		writeJSON(w, code, map[string]any{"status": "error", "message": "upload rejected: " + err.Error()})
		return
	}
	job, err := s.store.CommitUpload(jobID, slot, up)
	if err != nil {
		os.Remove(up.Path)
		clear(up.Key)
		uploadError(w, err)
		return
	}
	log.Printf("upload %s/%s stored: %d bytes in %s (%s)", jobID, slot, up.PlainSize,
		time.Since(started).Round(time.Millisecond), memoryUsage())
	if job.Status == jobs.StatusQueued {
		log.Printf("job %s all required files received — queued", jobID)
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"status": job.Status,
		"job_id": jobID,
		"bytes":  up.PlainSize,
	})
}

// uploadError maps a store refusal to its HTTP status.
func uploadError(w http.ResponseWriter, err error) {
	code := http.StatusBadRequest
	switch {
	case strings.Contains(err.Error(), "unknown job_id"),
		errors.Is(err, jobs.ErrUnknownSlot), errors.Is(err, jobs.ErrSlotNotAllowed):
		code = http.StatusNotFound
	case errors.Is(err, jobs.ErrUploadInProgress):
		code = http.StatusConflict
	}
	writeJSON(w, code, map[string]any{"status": "error", "message": err.Error()})
}

// writeError marks a failure to write the upload to disk (a server fault, not
// a bad upload).
type writeError struct{ err error }

func (e *writeError) Error() string { return "store upload: " + e.err.Error() }
func (e *writeError) Unwrap() error { return e.err }

// receiveUpload reads the stream header, unwraps its AES key, and writes the
// whole stream — header and chunks, unchanged — to a temp file in dir,
// checking the framing as it goes. On success the caller owns the file and
// the key.
func (s *Server) receiveUpload(body io.Reader, browserPubB64u, dir, slot string) (up jobs.Upload, err error) {
	hdr, rawHeader, err := chunked.ReadHeader(body)
	if err != nil {
		return up, err
	}
	key, err := s.unwrapUploadKey(hdr, browserPubB64u)
	if err != nil {
		return up, err
	}
	f, err := os.CreateTemp(dir, slot+"-*.part")
	if err != nil {
		clear(key)
		return up, &writeError{err}
	}
	defer func() {
		if err != nil {
			f.Close()
			os.Remove(f.Name())
			clear(key)
		}
	}()

	out := &recordingWriter{w: bufio.NewWriterSize(f, 1<<20)}
	if _, err := out.Write(rawHeader); err != nil {
		return up, &writeError{err}
	}
	plainSize, err := chunked.CopyChunks(body, hdr, out)
	if out.err != nil {
		return up, &writeError{out.err}
	}
	if err != nil {
		return up, err
	}
	if err := out.w.Flush(); err != nil {
		return up, &writeError{err}
	}
	if err := f.Sync(); err != nil {
		return up, &writeError{err}
	}
	if err := f.Close(); err != nil {
		return up, &writeError{err}
	}
	return jobs.Upload{Path: f.Name(), Key: key, PlainSHA256: hdr.PlaintextSHA256, PlainSize: plainSize}, nil
}

// unwrapUploadKey opens the stream's HPKE-wrapped AES-256 key with the
// Buffer's current HPKE key. It must happen at upload: HPKE keys rotate, so a
// wrapped key kept for later could become unopenable.
func (s *Server) unwrapUploadKey(hdr chunked.Header, browserPubB64u string) ([]byte, error) {
	if browserPubB64u == "" {
		return nil, fmt.Errorf("missing X-RATLS-Browser-HPKE header")
	}
	browserPub, err := base64.RawURLEncoding.DecodeString(browserPubB64u)
	if err != nil || len(browserPub) != 32 {
		return nil, fmt.Errorf("bad X-RATLS-Browser-HPKE")
	}
	enc, err := base64.StdEncoding.DecodeString(hdr.Enc)
	if err != nil {
		return nil, fmt.Errorf("decode enc: %w", err)
	}
	wrapped, err := base64.StdEncoding.DecodeString(hdr.WrappedKey)
	if err != nil {
		return nil, fmt.Errorf("decode wrapped_key: %w", err)
	}
	// No AAD — key wrapping uses info binding only.
	hpkePriv, hpkePub := s.builder.HPKEKeys()
	info := hpkepkg.ContextInfo(hpkePub.Bytes(), browserPub)
	key, err := hpkepkg.OpenFromBrowser(hpkePriv, enc, info, nil, wrapped)
	if err != nil {
		return nil, fmt.Errorf("hpke key unwrap: %w", err)
	}
	if len(key) != aesKeyLen {
		clear(key)
		return nil, fmt.Errorf("unwrapped key is %d bytes, want %d", len(key), aesKeyLen)
	}
	return key, nil
}

// recordingWriter remembers the first write failure, so a full disk is told
// apart from a bad upload.
type recordingWriter struct {
	w   *bufio.Writer
	err error
}

func (rw *recordingWriter) Write(p []byte) (int, error) {
	n, err := rw.w.Write(p)
	if err != nil && rw.err == nil {
		rw.err = err
	}
	return n, err
}

// checkDiskSpace refuses an upload that would not fit, with 10% to spare.
func checkDiskSpace(dir string, size int64) error {
	var st syscall.Statfs_t
	if err := syscall.Statfs(dir, &st); err != nil {
		log.Printf("upload: free disk space check skipped: %v", err)
		return nil
	}
	free := st.Bavail * uint64(st.Bsize)
	if need := uint64(size) + uint64(size)/10; free < need {
		return fmt.Errorf("the upload needs %d bytes of disk, %d free", need, free)
	}
	return nil
}

// processJob creates the job record in the in-process store. The response
// JSON shape is unchanged from the Flask era: {status, job_id, dataset_id}.
func (s *Server) processJob(plaintext []byte, keycloakSub string, keycloakToken string) []byte {
	var jobReq JobRequest
	if err := json.Unmarshal(plaintext, &jobReq); err != nil {
		return jsonErr("invalid job payload: " + err.Error())
	}
	if err := jobReq.validate(); err != nil {
		return jsonErr(err.Error())
	}

	job, err := s.store.Create(jobs.NewJobRequest{
		DatasetID:     jobReq.DatasetID,
		ModelFormat:   jobReq.ModelFormat,
		ModelSHA256:   jobReq.ModelSHA256,
		WeightsSHA256: jobReq.WeightsSHA256,
		AdaptorSHA:    jobReq.AdaptorSHA256,
		InputSpecSHA:  jobReq.InputSpecSHA256,
		SubmittedBy:   keycloakSub,
		KeycloakToken: keycloakToken,
	})
	if err != nil {
		return jsonErr(err.Error())
	}
	log.Printf("submit: job %s created (dataset %s, format %s)", job.JobID, job.DatasetID, job.ModelFormat)

	respBody, _ := json.Marshal(map[string]any{
		"status":     job.Status, // "pending_upload"
		"job_id":     job.JobID,
		"dataset_id": job.DatasetID,
	})
	return respBody
}

func jsonErr(msg string) []byte {
	b, _ := json.Marshal(map[string]string{"status": "error", "message": msg})
	return b
}
