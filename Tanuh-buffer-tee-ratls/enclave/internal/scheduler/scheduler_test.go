package scheduler

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"mime/multipart"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/datakaveri/tanuh-buffer-tee/internal/chunked"
	"github.com/datakaveri/tanuh-buffer-tee/internal/config"
	"github.com/datakaveri/tanuh-buffer-tee/internal/dispatch"
	"github.com/datakaveri/tanuh-buffer-tee/internal/jobs"
)

func sha(b []byte) string {
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}

// slotKey is the AES key the test "uploads" each slot with.
func slotKey(slot string) []byte { return bytes.Repeat([]byte(slot[:1]), 32) }

// queuedJob creates a job and stores each file as its upload: the stored
// stream (the store never looks inside) and a per-slot key.
func queuedJob(t *testing.T, format string, files map[string][]byte) (*Scheduler, *jobs.Job) {
	t.Helper()
	store, err := jobs.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	req := jobs.NewJobRequest{
		DatasetID:     "d8c50b6b-af58-4f9d-9af8-4d42bfb1c4bb",
		ModelFormat:   format,
		ModelSHA256:   sha(files[jobs.SlotModel]),
		AdaptorSHA:    sha(files[jobs.SlotAdaptor]),
		SubmittedBy:   "sub-123",
		KeycloakToken: "token",
	}
	if w, ok := files[jobs.SlotWeights]; ok {
		req.WeightsSHA256 = sha(w)
	}
	if spec, ok := files[jobs.SlotInputSpec]; ok {
		req.InputSpecSHA = sha(spec)
	}
	job, err := store.Create(req)
	if err != nil {
		t.Fatal(err)
	}
	for slot, data := range files {
		dir, err := store.BeginUpload(job.JobID, slot)
		if err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(dir, slot+".part")
		os.WriteFile(path, data, 0o600) //nolint:errcheck
		if job, err = store.CommitUpload(job.JobID, slot, jobs.Upload{Path: path, Key: slotKey(slot)}); err != nil {
			t.Fatal(err)
		}
		store.EndUpload(job.JobID, slot)
	}
	if job.Status != jobs.StatusQueued {
		t.Fatalf("job not queued: %s", job.Status)
	}
	return &Scheduler{Store: store, Cfg: config.Config{CallbackBase: "https://buffer:8443"}}, job
}

// sent is a body as the Processing TEE reads it: the metadata, then each
// part's bytes by name, in order.
type sent struct {
	meta  map[string]any
	order []string
	parts map[string][]byte
}

// send prepares and writes the job's payload, then parses it back.
func send(t *testing.T, s *Scheduler, job *jobs.Job) sent {
	t.Helper()
	p, err := s.preparePayload(job)
	if err != nil {
		t.Fatal(err)
	}
	defer p.clear()
	var body bytes.Buffer
	if err := p.write(&body); err != nil {
		t.Fatal(err)
	}
	mediaType, params, err := mime.ParseMediaType(p.contentType())
	if err != nil || mediaType != "multipart/form-data" {
		t.Fatalf("content type %q", p.contentType())
	}
	mr := multipart.NewReader(&body, params["boundary"])
	out := sent{parts: map[string][]byte{}}
	for {
		part, err := mr.NextRawPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		data, _ := io.ReadAll(part)
		if part.FormName() == "metadata" {
			if len(out.order) != 0 {
				t.Fatal("metadata is not the first part")
			}
			if err := json.Unmarshal(data, &out.meta); err != nil {
				t.Fatal(err)
			}
		}
		out.order = append(out.order, part.FormName())
		out.parts[part.FormName()] = data
	}
	return out
}

func TestPayloadV3Shape(t *testing.T) {
	files := map[string][]byte{
		jobs.SlotModel:   []byte("onnx-graph-ciphertext"),
		jobs.SlotWeights: []byte("onnx-weights-ciphertext"),
		jobs.SlotAdaptor: []byte("adaptor-ciphertext"),
	}
	s, job := queuedJob(t, jobs.FormatONNX, files)
	got := send(t, s, job)

	var keys []string
	for k := range got.meta {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	want := []string{"artifacts", "buffer_job_url", "dataset_id", "job_id", "keycloak_token",
		"model_format", "payload_version", "submitted_at_unix", "submitted_by"}
	if !reflect.DeepEqual(keys, want) {
		t.Fatalf("metadata keys\n got %v\nwant %v", keys, want)
	}
	if got.meta["payload_version"].(float64) != PayloadVersion || got.meta["model_format"] != jobs.FormatONNX {
		t.Fatalf("version/format: %v %v", got.meta["payload_version"], got.meta["model_format"])
	}
	if got.meta["buffer_job_url"] != "https://buffer:8443/v1/jobs/"+job.JobID {
		t.Fatalf("buffer_job_url %v", got.meta["buffer_job_url"])
	}
	if !reflect.DeepEqual(got.order, []string{"metadata", "model", "weights", "adaptor"}) {
		t.Fatalf("part order %v", got.order)
	}
	arts := got.meta["artifacts"].(map[string]any)
	for slot, data := range files {
		a := arts[slot].(map[string]any)
		key, _ := base64.StdEncoding.DecodeString(a["aes_key"].(string))
		if !bytes.Equal(got.parts[slot], data) {
			t.Errorf("%s: part is not the stored stream", slot)
		}
		if a["encoding"] != chunked.Encoding || a["sha256"] != sha(data) ||
			a["ciphertext_size"].(float64) != float64(len(data)) || !bytes.Equal(key, slotKey(slot)) {
			t.Errorf("%s: metadata %v", slot, a)
		}
	}
}

func TestPayloadOnlyCarriesTheFormatsArtifacts(t *testing.T) {
	s, job := queuedJob(t, jobs.FormatHuggingFace, map[string][]byte{
		jobs.SlotModel:   []byte("hf.zip"),
		jobs.SlotAdaptor: []byte("adaptor.py"),
	})
	got := send(t, s, job)
	if arts := got.meta["artifacts"].(map[string]any); len(arts) != 2 || arts[jobs.SlotWeights] != nil {
		t.Fatalf("artifacts = %v", arts)
	}
	if !reflect.DeepEqual(got.order, []string{"metadata", "model", "adaptor"}) {
		t.Fatalf("part order %v", got.order)
	}
}

func TestPayloadCarriesTheInputSpec(t *testing.T) {
	spec := []byte(`{"input_size": [256, 256], "resize": "bicubic"}`)
	s, job := queuedJob(t, jobs.FormatTorchScript, map[string][]byte{
		jobs.SlotModel:     []byte("model.pt"),
		jobs.SlotAdaptor:   []byte("adaptor.py"),
		jobs.SlotInputSpec: spec,
	})
	got := send(t, s, job)
	a := got.meta["artifacts"].(map[string]any)[jobs.SlotInputSpec].(map[string]any)
	if !bytes.Equal(got.parts[jobs.SlotInputSpec], spec) || a["sha256"] != sha(spec) {
		t.Fatalf("input_spec %v", a)
	}
}

// After a Buffer restart the upload keys are gone: the job fails before any
// VM is provisioned.
func TestPreparePayloadFailsWhenKeysAreLost(t *testing.T) {
	s, job := queuedJob(t, jobs.FormatTorchScript, map[string][]byte{
		jobs.SlotModel:   []byte("model.pt"),
		jobs.SlotAdaptor: []byte("adaptor.py"),
	})
	restarted, err := jobs.Open(filepath.Dir(filepath.Dir(filepath.Dir(job.ArtifactDir))))
	if err != nil {
		t.Fatal(err)
	}
	s.Store = restarted
	if _, err := s.preparePayload(job); err == nil || !strings.Contains(err.Error(), "resubmit") {
		t.Fatalf("err = %v, want the key-lost error", err)
	}
}

func TestPayloadWriteFailsIfAFileChangedSize(t *testing.T) {
	s, job := queuedJob(t, jobs.FormatTorchScript, map[string][]byte{
		jobs.SlotModel:   []byte("model.pt"),
		jobs.SlotAdaptor: []byte("adaptor.py"),
	})
	p, err := s.preparePayload(job)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(job.ArtifactDir, "model.enc"), []byte("truncated"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := p.write(io.Discard); err == nil || !strings.Contains(err.Error(), "send model") {
		t.Fatalf("err = %v, want a size mismatch on model", err)
	}
}

// A refusal the Processing TEE would repeat fails the job; a busy TEE or a
// network error leaves it queued for the next tick.
func TestDispatchFailureHandling(t *testing.T) {
	cases := []struct {
		name       string
		err        error
		wantStatus string
	}{
		{"bad payload", &dispatch.RejectedError{Status: 400, Message: "model does not match its submit-time SHA-256"}, jobs.StatusError},
		{"version mismatch", &dispatch.RejectedError{Status: 415, Message: "payload must be multipart/form-data (payload v3)"}, jobs.StatusError},
		{"no disk", &dispatch.RejectedError{Status: 507, Message: "artifacts need 5 bytes of disk, 1 free"}, jobs.StatusError},
		{"busy", &dispatch.RejectedError{Status: 409, Message: "Already running job job-1"}, jobs.StatusQueued},
		{"processing error", &dispatch.RejectedError{Status: 500, Message: "intake failed"}, jobs.StatusQueued},
		{"network", errors.New("dispatch: POST /api/load-model: connection reset"), jobs.StatusQueued},
	}
	for _, c := range cases {
		s, job := queuedJob(t, jobs.FormatTorchScript, map[string][]byte{
			jobs.SlotModel:   []byte("model.pt"),
			jobs.SlotAdaptor: []byte("adaptor.py"),
		})
		s.dispatchFailed(job, c.err)
		got, _ := s.Store.Get(job.JobID)
		if got.Status != c.wantStatus {
			t.Errorf("%s: status %s, want %s", c.name, got.Status, c.wantStatus)
		}
		if c.wantStatus == jobs.StatusError {
			if msg := got.CompletionError["error_message"]; msg != c.err.(*dispatch.RejectedError).Message {
				t.Errorf("%s: error message %v", c.name, msg)
			}
		} else if next, _ := s.Store.NextQueued(); next == nil || next.JobID != job.JobID {
			t.Errorf("%s: job left the queue", c.name)
		}
	}
}

func TestPayloadClearWipesTheKeys(t *testing.T) {
	s, job := queuedJob(t, jobs.FormatTorchScript, map[string][]byte{
		jobs.SlotModel:   []byte("model.pt"),
		jobs.SlotAdaptor: []byte("adaptor.py"),
	})
	p, err := s.preparePayload(job)
	if err != nil {
		t.Fatal(err)
	}
	key := base64.StdEncoding.EncodeToString(slotKey(jobs.SlotModel))
	if !bytes.Contains(p.metadata, []byte(key)) {
		t.Fatal("metadata does not carry the model key")
	}
	p.clear()
	if bytes.Contains(p.metadata, []byte(key)) {
		t.Fatal("key still in the metadata after clear")
	}
}
