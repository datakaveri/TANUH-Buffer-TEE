package jobs

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func sha(b []byte) string {
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}

var (
	modelBytes   = []byte("model-bytes")
	weightsBytes = []byte("weights-bytes")
	adaptorBytes = []byte("print('adaptor')")
)

func newJob(t *testing.T, format string, withWeights bool) (*Store, *Job) {
	t.Helper()
	s, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	req := NewJobRequest{
		DatasetID:   "99303f17-8be7-421d-9d75-81dd6cdb7c7b",
		ModelFormat: format,
		ModelSHA256: sha(modelBytes),
		AdaptorSHA:  sha(adaptorBytes),
	}
	if withWeights {
		req.WeightsSHA256 = sha(weightsBytes)
	}
	job, err := s.Create(req)
	if err != nil {
		t.Fatal(err)
	}
	return s, job
}

// receive runs an upload through the store as the upload handler does: begin,
// write the stream to a file in the returned directory, commit, end. data
// stands in for the encrypted stream (the store never looks inside it); its
// SHA-256 is sent as the stream header's plaintext_sha256.
func receive(t *testing.T, s *Store, id, slot string, data []byte) (*Job, error) {
	t.Helper()
	dir, err := s.BeginUpload(id, slot)
	if err != nil {
		return nil, err
	}
	defer s.EndUpload(id, slot)
	path := filepath.Join(dir, slot+".part")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	return s.CommitUpload(id, slot, Upload{Path: path, Key: bytes.Repeat([]byte{7}, 32), PlainSHA256: sha(data), PlainSize: int64(len(data))})
}

func mustReceive(t *testing.T, s *Store, id, slot string, data []byte) *Job {
	t.Helper()
	job, err := receive(t, s, id, slot, data)
	if err != nil {
		t.Fatalf("receive %s: %v", slot, err)
	}
	return job
}

func TestONNXWithWeightsQueuesOnlyWhenAllThreeArrive(t *testing.T) {
	s, job := newJob(t, FormatONNX, true)
	if got := mustReceive(t, s, job.JobID, SlotModel, modelBytes).Status; got != StatusPendingUpload {
		t.Fatalf("after model: %s", got)
	}
	if got := mustReceive(t, s, job.JobID, SlotAdaptor, adaptorBytes).Status; got != StatusPendingUpload {
		t.Fatalf("after adaptor: %s (weights were committed, so still pending)", got)
	}
	if got := mustReceive(t, s, job.JobID, SlotWeights, weightsBytes).Status; got != StatusQueued {
		t.Fatalf("after weights: %s", got)
	}
	if next, _ := s.NextQueued(); next == nil || next.JobID != job.JobID {
		t.Fatal("job not in the queue")
	}
}

func TestONNXWithoutWeightsQueuesOnModelAndAdaptor(t *testing.T) {
	s, job := newJob(t, FormatONNX, false)
	mustReceive(t, s, job.JobID, SlotAdaptor, adaptorBytes)
	if got := mustReceive(t, s, job.JobID, SlotModel, modelBytes).Status; got != StatusQueued {
		t.Fatalf("status %s, want queued", got)
	}
}

func TestUncommittedWeightsRejected(t *testing.T) {
	s, job := newJob(t, FormatONNX, false)
	if _, err := receive(t, s, job.JobID, SlotWeights, weightsBytes); err == nil ||
		!strings.Contains(err.Error(), "no SHA-256 was committed") {
		t.Fatalf("uncommitted upload accepted: %v", err)
	}
}

func TestWeightsSlotNotAllowedForTorchScriptOrHF(t *testing.T) {
	for _, f := range []string{FormatTorchScript, FormatHuggingFace} {
		s, job := newJob(t, f, false)
		if _, err := receive(t, s, job.JobID, SlotWeights, weightsBytes); !errors.Is(err, ErrSlotNotAllowed) {
			t.Fatalf("%s: weights slot err=%v, want ErrSlotNotAllowed", f, err)
		}
		mustReceive(t, s, job.JobID, SlotModel, modelBytes)
		if got := mustReceive(t, s, job.JobID, SlotAdaptor, adaptorBytes).Status; got != StatusQueued {
			t.Fatalf("%s: status %s, want queued with model + adaptor", f, got)
		}
	}
}

func TestInputSpecSlot(t *testing.T) {
	spec := []byte(`{"input_size": [224, 224]}`)
	s, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	job, err := s.Create(NewJobRequest{ModelFormat: FormatTorchScript, ModelSHA256: sha(modelBytes),
		AdaptorSHA: sha(adaptorBytes), InputSpecSHA: sha(spec)})
	if err != nil {
		t.Fatal(err)
	}
	mustReceive(t, s, job.JobID, SlotModel, modelBytes)
	if got := mustReceive(t, s, job.JobID, SlotAdaptor, adaptorBytes).Status; got != StatusPendingUpload {
		t.Fatalf("after model + adaptor: %s (input spec was committed, so still pending)", got)
	}
	if got := mustReceive(t, s, job.JobID, SlotInputSpec, spec).Status; got != StatusQueued {
		t.Fatalf("after input spec: %s", got)
	}
	if a, err := s.Artifact(job.JobID, SlotInputSpec); err != nil {
		t.Fatal(err)
	} else if got, _ := os.ReadFile(a.Path); !bytes.Equal(got, spec) {
		t.Fatalf("stored input spec %q", got)
	}

	// Not committed: refused; with Hugging Face: not a slot of that format.
	s2, plain := newJob(t, FormatTorchScript, false)
	if _, err := receive(t, s2, plain.JobID, SlotInputSpec, spec); err == nil {
		t.Fatal("uncommitted input spec accepted")
	}
	s3, hf := newJob(t, FormatHuggingFace, false)
	if _, err := receive(t, s3, hf.JobID, SlotInputSpec, spec); !errors.Is(err, ErrSlotNotAllowed) {
		t.Fatalf("input spec on Hugging Face err=%v, want ErrSlotNotAllowed", err)
	}
}

func TestUnknownSlotRejected(t *testing.T) {
	s, job := newJob(t, FormatONNX, false)
	for _, slot := range []string{"preprocessing", "model.onnx", "../job.json", ""} {
		if _, err := receive(t, s, job.JobID, slot, adaptorBytes); !errors.Is(err, ErrUnknownSlot) {
			t.Fatalf("slot %q: err=%v, want ErrUnknownSlot", slot, err)
		}
	}
}

func TestHashMismatchRejectedAndCaseInsensitiveMatchAccepted(t *testing.T) {
	s, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	job, err := s.Create(NewJobRequest{
		ModelFormat: FormatTorchScript,
		ModelSHA256: strings.ToUpper(sha(modelBytes)), // browser sent upper-case hex
		AdaptorSHA:  sha(adaptorBytes),
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := receive(t, s, job.JobID, SlotModel, []byte("swapped")); err == nil ||
		!strings.Contains(err.Error(), "SHA256 mismatch") {
		t.Fatalf("swapped model accepted: %v", err)
	}
	mustReceive(t, s, job.JobID, SlotModel, modelBytes)
}

// The Buffer never decrypts, so a stream whose header carries no hash is
// stored; the Processing TEE checks the content against the commitment.
func TestUploadWithoutHeaderHashAccepted(t *testing.T) {
	s, job := newJob(t, FormatONNX, false)
	dir, err := s.BeginUpload(job.JobID, SlotModel)
	if err != nil {
		t.Fatal(err)
	}
	defer s.EndUpload(job.JobID, SlotModel)
	path := filepath.Join(dir, "model.part")
	os.WriteFile(path, []byte("ciphertext"), 0o600) //nolint:errcheck
	if _, err := s.CommitUpload(job.JobID, SlotModel, Upload{Path: path, Key: make([]byte, 32)}); err != nil {
		t.Fatal(err)
	}
}

func TestNoFileSizeCap(t *testing.T) {
	big := bytes.Repeat([]byte("x"), (1<<20)+1) // over the old 1 MiB adaptor limit
	s, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	job, _ := s.Create(NewJobRequest{ModelFormat: FormatONNX, ModelSHA256: sha(modelBytes), AdaptorSHA: sha(big)})
	mustReceive(t, s, job.JobID, SlotAdaptor, big)
}

func TestConcurrentUploadToSameSlotRejected(t *testing.T) {
	s, job := newJob(t, FormatONNX, false)
	if _, err := s.BeginUpload(job.JobID, SlotModel); err != nil {
		t.Fatal(err)
	}
	if _, err := s.BeginUpload(job.JobID, SlotModel); !errors.Is(err, ErrUploadInProgress) {
		t.Fatalf("second upload err=%v, want ErrUploadInProgress", err)
	}
	if _, err := s.BeginUpload(job.JobID, SlotAdaptor); err != nil {
		t.Fatalf("another slot blocked: %v", err)
	}
	s.EndUpload(job.JobID, SlotModel)
	if _, err := s.BeginUpload(job.JobID, SlotModel); err != nil {
		t.Fatalf("slot still blocked after EndUpload: %v", err)
	}
}

func TestNoUploadsAfterQueued(t *testing.T) {
	s, job := newJob(t, FormatTorchScript, false)
	mustReceive(t, s, job.JobID, SlotModel, modelBytes)
	mustReceive(t, s, job.JobID, SlotAdaptor, adaptorBytes)
	if _, err := receive(t, s, job.JobID, SlotModel, modelBytes); err == nil {
		t.Fatal("re-upload after queueing accepted")
	}
}

func TestArtifactBySlot(t *testing.T) {
	s, job := newJob(t, FormatONNX, false)
	mustReceive(t, s, job.JobID, SlotModel, modelBytes)
	a, err := s.Artifact(job.JobID, SlotModel)
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(a.Path); !bytes.Equal(got, modelBytes) || a.Size != int64(len(modelBytes)) ||
		filepath.Base(a.Path) != "model.enc" || len(a.Key) != 32 {
		t.Fatalf("artifact = %+v", a)
	}
	clear(a.Key) // the caller's copy; the store's key is unaffected
	if again, _ := s.Artifact(job.JobID, SlotModel); again.Key[0] != 7 {
		t.Fatal("clearing the returned key cleared the stored one")
	}
	if _, err := s.Artifact(job.JobID, "preprocessing"); !errors.Is(err, ErrUnknownSlot) {
		t.Fatalf("unknown slot read: %v", err)
	}
}

// Upload keys live only in memory: never in job.json, gone once the job is
// finished, and gone after a restart (a new Store over the same directory).
func TestUploadKeysOnlyInMemory(t *testing.T) {
	root := t.TempDir()
	s, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	job, _ := s.Create(NewJobRequest{ModelFormat: FormatONNX, ModelSHA256: sha(modelBytes), AdaptorSHA: sha(adaptorBytes)})
	mustReceive(t, s, job.JobID, SlotModel, modelBytes)
	mustReceive(t, s, job.JobID, SlotAdaptor, adaptorBytes)

	raw, _ := os.ReadFile(filepath.Join(root, "jobs", job.JobID, "job.json"))
	if bytes.Contains(raw, []byte("BwcHBwcH")) || bytes.Contains(raw, bytes.Repeat([]byte{7}, 8)) {
		t.Fatal("job.json holds an upload key")
	}

	restarted, _ := Open(root)
	if _, err := restarted.Artifact(job.JobID, SlotModel); err == nil || !strings.Contains(err.Error(), "resubmit") {
		t.Fatalf("after restart err=%v, want the key-lost error", err)
	}

	if _, err := s.Fail(job.JobID, "test", nil); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Artifact(job.JobID, SlotModel); err == nil {
		t.Fatal("key still held after the job failed")
	}
}
