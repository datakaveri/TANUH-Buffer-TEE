package jobs

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
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

func mustReceive(t *testing.T, s *Store, id, slot string, data []byte) *Job {
	t.Helper()
	job, err := s.ReceiveArtifact(id, slot, data)
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
	if _, err := s.ReceiveArtifact(job.JobID, SlotWeights, weightsBytes); err == nil ||
		!strings.Contains(err.Error(), "no SHA-256 was committed") {
		t.Fatalf("uncommitted upload accepted: %v", err)
	}
}

func TestWeightsSlotNotAllowedForTorchScriptOrHF(t *testing.T) {
	for _, f := range []string{FormatTorchScript, FormatHuggingFace} {
		s, job := newJob(t, f, false)
		if _, err := s.ReceiveArtifact(job.JobID, SlotWeights, weightsBytes); !errors.Is(err, ErrSlotNotAllowed) {
			t.Fatalf("%s: weights slot err=%v, want ErrSlotNotAllowed", f, err)
		}
		mustReceive(t, s, job.JobID, SlotModel, modelBytes)
		if got := mustReceive(t, s, job.JobID, SlotAdaptor, adaptorBytes).Status; got != StatusQueued {
			t.Fatalf("%s: status %s, want queued with model + adaptor", f, got)
		}
	}
}

func TestUnknownSlotRejected(t *testing.T) {
	s, job := newJob(t, FormatONNX, false)
	for _, slot := range []string{"preprocessing", "model.onnx", "../job.json", ""} {
		if _, err := s.ReceiveArtifact(job.JobID, slot, adaptorBytes); !errors.Is(err, ErrUnknownSlot) {
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
	if _, err := s.ReceiveArtifact(job.JobID, SlotModel, []byte("swapped")); err == nil ||
		!strings.Contains(err.Error(), "SHA256 mismatch") {
		t.Fatalf("swapped model accepted: %v", err)
	}
	mustReceive(t, s, job.JobID, SlotModel, modelBytes)
}

func TestAdaptorSizeLimit(t *testing.T) {
	big := bytes.Repeat([]byte("x"), (1<<20)+1)
	s, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	job, _ := s.Create(NewJobRequest{ModelFormat: FormatONNX, ModelSHA256: sha(modelBytes), AdaptorSHA: sha(big)})
	if _, err := s.ReceiveArtifact(job.JobID, SlotAdaptor, big); err == nil || !strings.Contains(err.Error(), "limit") {
		t.Fatalf("oversized adaptor accepted: %v", err)
	}
}

func TestNoUploadsAfterQueued(t *testing.T) {
	s, job := newJob(t, FormatTorchScript, false)
	mustReceive(t, s, job.JobID, SlotModel, modelBytes)
	mustReceive(t, s, job.JobID, SlotAdaptor, adaptorBytes)
	if _, err := s.ReceiveArtifact(job.JobID, SlotModel, modelBytes); err == nil {
		t.Fatal("re-upload after queueing accepted")
	}
}

func TestArtifactBytesBySlot(t *testing.T) {
	s, job := newJob(t, FormatONNX, false)
	mustReceive(t, s, job.JobID, SlotModel, modelBytes)
	got, err := s.ArtifactBytes(job.JobID, SlotModel)
	if err != nil || !bytes.Equal(got, modelBytes) {
		t.Fatalf("ArtifactBytes = %q, %v", got, err)
	}
	if _, err := s.ArtifactBytes(job.JobID, "preprocessing"); !errors.Is(err, ErrUnknownSlot) {
		t.Fatalf("unknown slot read: %v", err)
	}
}
