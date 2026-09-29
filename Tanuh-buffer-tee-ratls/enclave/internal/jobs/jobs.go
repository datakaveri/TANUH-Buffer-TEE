// Package jobs is the buffer's persistent job store: records, artifacts,
// the ordered queue, and dispatch state. The on-disk layout is kept
// byte-compatible with the former Python manager (jobs/<id>/job.json,
// queue.json, runtime/dispatch_state.json, outgoing/*-secure-dispatch.json)
// so a rollback to the Python image reads the same state and the same
// debugging tooling keeps working.
package jobs

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	StatusPendingUpload = "pending_upload"
	StatusQueued        = "queued"
	StatusDispatched    = "dispatched"
	StatusComplete      = "complete"
	StatusError         = "error"
)

// Delivery mirrors the Python "delivery" sub-object on dispatched jobs.
type Delivery struct {
	Mode        string `json:"mode"`
	PayloadPath string `json:"payload_path"`
}

// Job is the persisted job record. Field names match the Python manager's
// JSON exactly — do not rename tags.
type Job struct {
	JobID              string         `json:"job_id"`
	Status             string         `json:"status"`
	DatasetID          string         `json:"dataset_id"`
	SubmittedAtUnix    int64          `json:"submitted_at_unix"`
	UpdatedAtUnix      int64          `json:"updated_at_unix"`
	Hyperparameters    map[string]any `json:"hyperparameters"`
	ModelFormat        string         `json:"model_format"` // onnx | torchscript | huggingface
	ArtifactDir        string         `json:"artifact_dir"` // one file per upload slot (see artifacts.go)
	SubmittedBy        string         `json:"submitted_by"`
	KeycloakToken      string         `json:"keycloak_token"`
	ModelSHA256        string         `json:"model_sha256_expected"`
	WeightsSHA256      string         `json:"weights_sha256_expected"`
	AdaptorSHA         string         `json:"adaptor_sha256_expected"`
	Notes              string         `json:"notes"`
	GPUAttempts        int            `json:"gpu_provision_attempts,omitempty"`
	ProvisioningTarget string         `json:"provisioning_target,omitempty"`
	AssignedTEE        string         `json:"assigned_processing_tee,omitempty"`
	DispatchedAtUnix   int64          `json:"dispatched_at_unix,omitempty"`
	Delivery           *Delivery      `json:"delivery,omitempty"`
	CompletedAtUnix    int64          `json:"completed_at_unix,omitempty"`
	CompletionSource   string         `json:"completion_source,omitempty"`
	CompletionError    map[string]any `json:"completion_error,omitempty"`
	RequeueCount       int            `json:"requeue_count,omitempty"`
}

// Store owns the on-disk job state. All mutation goes through one mutex —
// job volume is single-digit concurrent, matching the Python file-lock model.
type Store struct {
	mu   sync.Mutex
	root string // <base>/cvm_workflow/buffer
}

// Open creates the directory layout and returns the store.
func Open(root string) (*Store, error) {
	s := &Store{root: root}
	for _, d := range []string{
		root,
		s.jobsDir(),
		filepath.Join(root, "outgoing"),
		filepath.Join(root, "runtime"),
		filepath.Join(root, "shared_artifacts"),
	} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return nil, err
		}
	}
	return s, nil
}

func (s *Store) jobsDir() string   { return filepath.Join(s.root, "jobs") }
func (s *Store) queuePath() string { return filepath.Join(s.root, "queue.json") }
func (s *Store) dispatchStatePath() string {
	return filepath.Join(s.root, "runtime", "dispatch_state.json")
}
func (s *Store) jobDir(jobID string) string  { return filepath.Join(s.jobsDir(), jobID) }
func (s *Store) jobPath(jobID string) string { return filepath.Join(s.jobDir(jobID), "job.json") }

// OutgoingPayloadPath is where the secure dispatch payload for a job is staged.
func (s *Store) OutgoingPayloadPath(jobID string) string {
	return filepath.Join(s.root, "outgoing", jobID+"-secure-dispatch.json")
}

// ResultsPath is where per-job results would land (leaderboard-only today;
// kept for the /v1/results contract).
func (s *Store) ResultsPath(jobID string) string {
	return filepath.Join(s.jobDir(jobID), "results.json")
}

// ── Job lifecycle ─────────────────────────────────────────────────────────────

// NewJobRequest carries the validated fields from /v1/submit.
type NewJobRequest struct {
	DatasetID       string
	ModelFormat     string
	ModelSHA256     string
	WeightsSHA256   string
	AdaptorSHA      string
	SubmittedBy     string
	KeycloakToken   string
	Hyperparameters map[string]any
	Notes           string
}

// Create makes a new pending_upload job record.
func (s *Store) Create(req NewJobRequest) (*Job, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	jobID := "job-" + randHex(6)
	dir := s.jobDir(jobID)
	if _, err := os.Stat(dir); err == nil {
		return nil, fmt.Errorf("job %s already exists", jobID)
	}
	if err := os.MkdirAll(filepath.Join(dir, "artifacts"), 0o755); err != nil {
		return nil, err
	}

	now := time.Now().Unix()
	hp := req.Hyperparameters
	if hp == nil {
		hp = map[string]any{}
	}
	job := &Job{
		JobID:           jobID,
		Status:          StatusPendingUpload,
		DatasetID:       req.DatasetID,
		SubmittedAtUnix: now,
		UpdatedAtUnix:   now,
		Hyperparameters: hp,
		ModelFormat:     req.ModelFormat,
		ArtifactDir:     filepath.Join(dir, "artifacts"),
		SubmittedBy:     orDefault(req.SubmittedBy, "unknown"),
		KeycloakToken:   req.KeycloakToken,
		ModelSHA256:     strings.ToLower(req.ModelSHA256),
		WeightsSHA256:   strings.ToLower(req.WeightsSHA256),
		AdaptorSHA:      strings.ToLower(req.AdaptorSHA),
		Notes:           req.Notes,
	}
	if err := s.saveLocked(job); err != nil {
		return nil, err
	}
	return job, nil
}

// ReceiveArtifact stores an uploaded artifact for one slot, verifies it
// against the SHA-256 committed at submit (fail closed: a slot with no
// commitment accepts nothing), and queues the job once every expected slot is
// present. Returns the updated job.
func (s *Store) ReceiveArtifact(jobID, slot string, data []byte) (*Job, error) {
	spec, ok := ArtifactBySlot(slot)
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrUnknownSlot, slot)
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	job, err := s.getLocked(jobID)
	if err != nil {
		return nil, err
	}
	if job.Status != StatusPendingUpload {
		return nil, fmt.Errorf("job %s is not awaiting upload (status=%s)", jobID, job.Status)
	}
	if !spec.AllowedFor(job.ModelFormat) {
		return nil, fmt.Errorf("%w: %s for %s", ErrSlotNotAllowed, slot, job.ModelFormat)
	}
	expected := spec.Commitment(job)
	if expected == "" {
		return nil, fmt.Errorf("no SHA-256 was committed for %s at submit", slot)
	}
	if int64(len(data)) > spec.MaxBytes {
		return nil, fmt.Errorf("%s is %d bytes, over the %d-byte limit", slot, len(data), spec.MaxBytes)
	}
	sum := sha256.Sum256(data)
	if hex.EncodeToString(sum[:]) != expected {
		return nil, fmt.Errorf("SHA256 mismatch for %s", slot)
	}

	if err := os.MkdirAll(job.ArtifactDir, 0o755); err != nil {
		return nil, err
	}
	if err := os.WriteFile(filepath.Join(job.ArtifactDir, slot), data, 0o644); err != nil {
		return nil, err
	}
	job.UpdatedAtUnix = time.Now().Unix()

	required := true
	for _, a := range Artifacts() {
		if a.Expected(job) && !s.artifactExistsLocked(job, a.Slot) {
			required = false
			break
		}
	}
	if required {
		job.Status = StatusQueued
		if err := s.queueAppendLocked(jobID); err != nil {
			return nil, err
		}
	}
	if err := s.saveLocked(job); err != nil {
		return nil, err
	}
	return job, nil
}

func (s *Store) artifactExistsLocked(job *Job, name string) bool {
	_, err := os.Stat(filepath.Join(job.ArtifactDir, name))
	return err == nil
}

// ArtifactBytes returns a stored artifact's contents by slot.
func (s *Store) ArtifactBytes(jobID, slot string) ([]byte, error) {
	if _, ok := ArtifactBySlot(slot); !ok {
		return nil, fmt.Errorf("%w: %s", ErrUnknownSlot, slot)
	}
	s.mu.Lock()
	job, err := s.getLocked(jobID)
	s.mu.Unlock()
	if err != nil {
		return nil, err
	}
	return os.ReadFile(filepath.Join(job.ArtifactDir, slot))
}

// Get returns one job.
func (s *Store) Get(jobID string) (*Job, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.getLocked(jobID)
}

// List returns all jobs sorted by job dir name plus the queued ids.
func (s *Store) List() ([]*Job, []string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	jobs, err := s.allLocked()
	if err != nil {
		return nil, nil, err
	}
	return jobs, s.readQueueLocked(), nil
}

// Save persists mutations made by the scheduler/provisioner (attempt counts,
// provisioning target).
func (s *Store) Save(job *Job) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	job.UpdatedAtUnix = time.Now().Unix()
	return s.saveLocked(job)
}

// NextQueued returns the first queue entry whose record is still queued.
func (s *Store) NextQueued() (*Job, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, id := range s.readQueueLocked() {
		job, err := s.getLocked(id)
		if err != nil {
			continue
		}
		if job.Status == StatusQueued {
			return job, nil
		}
	}
	return nil, nil
}

// AnyDispatched reports whether any job is currently dispatched.
func (s *Store) AnyDispatched() (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	jobs, err := s.allLocked()
	if err != nil {
		return false, err
	}
	for _, j := range jobs {
		if j.Status == StatusDispatched {
			return true, nil
		}
	}
	return false, nil
}

// Dispatched returns all currently dispatched jobs.
func (s *Store) Dispatched() ([]*Job, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	jobs, err := s.allLocked()
	if err != nil {
		return nil, err
	}
	var out []*Job
	for _, j := range jobs {
		if j.Status == StatusDispatched {
			out = append(out, j)
		}
	}
	return out, nil
}

// MarkDispatched transitions queued → dispatched and removes the job from
// the queue.
func (s *Store) MarkDispatched(jobID, payloadPath, addr string) (*Job, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	job, err := s.getLocked(jobID)
	if err != nil {
		return nil, err
	}
	now := time.Now().Unix()
	job.Status = StatusDispatched
	job.AssignedTEE = addr
	job.DispatchedAtUnix = now
	job.UpdatedAtUnix = now
	job.Delivery = &Delivery{Mode: "ratls_json_payload", PayloadPath: payloadPath}
	if err := s.saveLocked(job); err != nil {
		return nil, err
	}
	s.queueRemoveLocked(jobID)
	return job, nil
}

// Complete records a terminal status from the Processing TEE callback.
// Only dispatched jobs can complete; anything else is a conflict.
func (s *Store) Complete(jobID string, succeeded bool, errInfo map[string]any) (*Job, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	job, err := s.getLocked(jobID)
	if err != nil {
		return nil, err
	}
	if job.Status != StatusDispatched {
		return job, fmt.Errorf("job is '%s', not dispatched", job.Status)
	}
	now := time.Now().Unix()
	if succeeded {
		job.Status = StatusComplete
	} else {
		job.Status = StatusError
	}
	job.CompletedAtUnix = now
	job.UpdatedAtUnix = now
	job.CompletionSource = "processing_tee_callback"
	job.CompletionError = errInfo
	if err := s.saveLocked(job); err != nil {
		return nil, err
	}
	s.queueRemoveLocked(jobID)
	return job, nil
}

// Requeue returns a stuck dispatched job to the queue (crash recovery),
// resetting its provisioning attempts — it provisioned fine before, so it
// gets a fresh GPU-first cycle. Returns the new requeue count.
func (s *Store) Requeue(jobID string) (*Job, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	job, err := s.getLocked(jobID)
	if err != nil {
		return nil, err
	}
	now := time.Now().Unix()
	job.Status = StatusQueued
	job.RequeueCount++
	job.GPUAttempts = 0
	job.AssignedTEE = ""
	job.DispatchedAtUnix = 0
	job.UpdatedAtUnix = now
	if err := s.saveLocked(job); err != nil {
		return nil, err
	}
	if err := s.queueAppendLocked(jobID); err != nil {
		return nil, err
	}
	return job, nil
}

// Fail marks a job terminally failed from the buffer side (e.g. requeue
// budget exhausted).
func (s *Store) Fail(jobID, source string, errInfo map[string]any) (*Job, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	job, err := s.getLocked(jobID)
	if err != nil {
		return nil, err
	}
	now := time.Now().Unix()
	job.Status = StatusError
	job.CompletedAtUnix = now
	job.UpdatedAtUnix = now
	job.CompletionSource = source
	job.CompletionError = errInfo
	if err := s.saveLocked(job); err != nil {
		return nil, err
	}
	s.queueRemoveLocked(jobID)
	return job, nil
}

// CleanupOld removes terminal jobs older than retention. Returns removed ids.
func (s *Store) CleanupOld(retention time.Duration) []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	jobs, err := s.allLocked()
	if err != nil {
		return nil
	}
	var removed []string
	now := time.Now().Unix()
	for _, j := range jobs {
		switch j.Status {
		case StatusComplete, StatusError, "rejected":
		default:
			continue
		}
		if now-j.UpdatedAtUnix < int64(retention.Seconds()) {
			continue
		}
		if err := os.RemoveAll(s.jobDir(j.JobID)); err == nil {
			removed = append(removed, j.JobID)
		}
		_ = os.Remove(s.OutgoingPayloadPath(j.JobID))
	}
	return removed
}

// StalledQueued returns queued jobs older than threshold (for warnings).
func (s *Store) StalledQueued(threshold time.Duration) []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []string
	now := time.Now().Unix()
	for _, id := range s.readQueueLocked() {
		job, err := s.getLocked(id)
		if err != nil || job.Status != StatusQueued {
			continue
		}
		if now-job.SubmittedAtUnix >= int64(threshold.Seconds()) {
			out = append(out, id)
		}
	}
	return out
}

// ── Dispatch state (runtime/dispatch_state.json) ──────────────────────────────

// UpdateDispatchState merges kv into the persisted dispatch-state map.
func (s *Store) UpdateDispatchState(kv map[string]any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	state := map[string]any{}
	if raw, err := os.ReadFile(s.dispatchStatePath()); err == nil {
		_ = json.Unmarshal(raw, &state)
	}
	for k, v := range kv {
		state[k] = v
	}
	if raw, err := json.MarshalIndent(state, "", "  "); err == nil {
		_ = os.WriteFile(s.dispatchStatePath(), raw, 0o644)
	}
}

// DispatchState returns the persisted dispatch-state map.
func (s *Store) DispatchState() map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	state := map[string]any{}
	if raw, err := os.ReadFile(s.dispatchStatePath()); err == nil {
		_ = json.Unmarshal(raw, &state)
	}
	return state
}

// ── internals ─────────────────────────────────────────────────────────────────

func (s *Store) getLocked(jobID string) (*Job, error) {
	raw, err := os.ReadFile(s.jobPath(jobID))
	if err != nil {
		return nil, fmt.Errorf("unknown job_id")
	}
	var job Job
	if err := json.Unmarshal(raw, &job); err != nil {
		return nil, fmt.Errorf("corrupt job record %s: %w", jobID, err)
	}
	return &job, nil
}

func (s *Store) allLocked() ([]*Job, error) {
	entries, err := os.ReadDir(s.jobsDir())
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	jobs := make([]*Job, 0, len(names))
	for _, n := range names {
		if job, err := s.getLocked(n); err == nil {
			jobs = append(jobs, job)
		}
	}
	return jobs, nil
}

func (s *Store) saveLocked(job *Job) error {
	raw, err := json.MarshalIndent(job, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(s.jobDir(job.JobID), 0o755); err != nil {
		return err
	}
	return os.WriteFile(s.jobPath(job.JobID), raw, 0o644)
}

func (s *Store) readQueueLocked() []string {
	raw, err := os.ReadFile(s.queuePath())
	if err != nil {
		return nil
	}
	var q struct {
		QueuedJobIDs []string `json:"queued_job_ids"`
	}
	if err := json.Unmarshal(raw, &q); err != nil {
		return nil
	}
	return q.QueuedJobIDs
}

func (s *Store) writeQueueLocked(ids []string) error {
	if ids == nil {
		ids = []string{}
	}
	raw, err := json.MarshalIndent(map[string][]string{"queued_job_ids": ids}, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(s.queuePath(), raw, 0o644)
}

func (s *Store) queueAppendLocked(jobID string) error {
	ids := s.readQueueLocked()
	for _, id := range ids {
		if id == jobID {
			return nil
		}
	}
	return s.writeQueueLocked(append(ids, jobID))
}

func (s *Store) queueRemoveLocked(jobID string) {
	ids := s.readQueueLocked()
	out := ids[:0]
	for _, id := range ids {
		if id != jobID {
			out = append(out, id)
		}
	}
	_ = s.writeQueueLocked(out)
}

func randHex(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		panic(err) // crypto/rand failure is unrecoverable
	}
	return hex.EncodeToString(b)
}

func orDefault(v, def string) string {
	if v == "" {
		return def
	}
	return v
}
