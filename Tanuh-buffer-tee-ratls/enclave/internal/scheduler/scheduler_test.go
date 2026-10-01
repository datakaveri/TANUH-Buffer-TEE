package scheduler

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/datakaveri/tanuh-buffer-tee/internal/config"
	"github.com/datakaveri/tanuh-buffer-tee/internal/jobs"
)

func sha(b []byte) string {
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}

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
		if job, err = store.ReceiveArtifact(job.JobID, slot, data); err != nil {
			t.Fatal(err)
		}
	}
	if job.Status != jobs.StatusQueued {
		t.Fatalf("job not queued: %s", job.Status)
	}
	return &Scheduler{Store: store, Cfg: config.Config{CallbackBase: "https://buffer:8443"}}, job
}

func decode(t *testing.T, raw []byte) map[string]any {
	t.Helper()
	var p map[string]any
	if err := json.Unmarshal(raw, &p); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestBuildPayloadV2Shape(t *testing.T) {
	files := map[string][]byte{
		jobs.SlotModel:   []byte("onnx-graph"),
		jobs.SlotWeights: []byte("onnx-weights"),
		jobs.SlotAdaptor: []byte("adaptor.py"),
	}
	s, job := queuedJob(t, jobs.FormatONNX, files)
	raw, err := s.buildPayload(job)
	if err != nil {
		t.Fatal(err)
	}
	p := decode(t, raw)

	var keys []string
	for k := range p {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	want := []string{"artifacts", "buffer_job_url", "dataset_id", "job_id", "keycloak_token",
		"model_format", "payload_version", "submitted_at_unix", "submitted_by"}
	if !reflect.DeepEqual(keys, want) {
		t.Fatalf("payload keys\n got %v\nwant %v", keys, want)
	}
	if p["payload_version"].(float64) != PayloadVersion || p["model_format"] != jobs.FormatONNX {
		t.Fatalf("version/format: %v %v", p["payload_version"], p["model_format"])
	}
	if p["buffer_job_url"] != "https://buffer:8443/v1/jobs/"+job.JobID {
		t.Fatalf("buffer_job_url %v", p["buffer_job_url"])
	}
	arts := p["artifacts"].(map[string]any)
	for slot, data := range files {
		a := arts[slot].(map[string]any)
		got, _ := base64.StdEncoding.DecodeString(a["base64"].(string))
		if string(got) != string(data) || a["sha256"] != sha(data) {
			t.Fatalf("artifact %s mismatch", slot)
		}
	}
}

func TestBuildPayloadOnlyCarriesTheFormatsArtifacts(t *testing.T) {
	s, job := queuedJob(t, jobs.FormatHuggingFace, map[string][]byte{
		jobs.SlotModel:   []byte("hf.zip"),
		jobs.SlotAdaptor: []byte("adaptor.py"),
	})
	raw, err := s.buildPayload(job)
	if err != nil {
		t.Fatal(err)
	}
	arts := decode(t, raw)["artifacts"].(map[string]any)
	if len(arts) != 2 || arts[jobs.SlotWeights] != nil {
		t.Fatalf("artifacts = %v", arts)
	}
}

func TestBuildPayloadCarriesTheInputSpec(t *testing.T) {
	spec := []byte(`{"input_size": [256, 256], "resize": "bicubic"}`)
	s, job := queuedJob(t, jobs.FormatTorchScript, map[string][]byte{
		jobs.SlotModel:     []byte("model.pt"),
		jobs.SlotAdaptor:   []byte("adaptor.py"),
		jobs.SlotInputSpec: spec,
	})
	raw, err := s.buildPayload(job)
	if err != nil {
		t.Fatal(err)
	}
	a := decode(t, raw)["artifacts"].(map[string]any)[jobs.SlotInputSpec].(map[string]any)
	got, _ := base64.StdEncoding.DecodeString(a["base64"].(string))
	if string(got) != string(spec) || a["sha256"] != sha(spec) {
		t.Fatalf("input_spec artifact %v", a)
	}
}

func TestBuildPayloadRejectsArtifactChangedOnDisk(t *testing.T) {
	s, job := queuedJob(t, jobs.FormatTorchScript, map[string][]byte{
		jobs.SlotModel:   []byte("model.pt"),
		jobs.SlotAdaptor: []byte("adaptor.py"),
	})
	if err := os.WriteFile(filepath.Join(job.ArtifactDir, jobs.SlotModel), []byte("swapped"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := s.buildPayload(job); err == nil || !strings.Contains(err.Error(), "submit-time SHA-256") {
		t.Fatalf("tampered artifact dispatched: %v", err)
	}
}
