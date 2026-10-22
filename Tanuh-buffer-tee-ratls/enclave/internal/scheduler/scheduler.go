// Package scheduler runs the buffer's background loop: retention cleanup,
// queue-stall warnings, crash recovery for dispatched jobs that never called
// back, and the dispatch cycle (check payload → provision → RA-TLS stream).
//
// Completion is callback-driven (the Processing TEE POSTs its terminal
// status to /v1/jobs/{id}/complete). A dispatched job is requeued ONLY when
// the dispatch timeout elapses with no callback AND its TEE is no longer
// healthy — and at most MaxRequeue times, after which it is failed. This
// replaces both the fire-and-forget finalizer and the old unbounded
// requeue-on-timeout that caused the re-run loop.
package scheduler

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/textproto"
	"os"
	"time"

	"github.com/datakaveri/tanuh-buffer-tee/internal/chunked"
	"github.com/datakaveri/tanuh-buffer-tee/internal/config"
	"github.com/datakaveri/tanuh-buffer-tee/internal/dispatch"
	"github.com/datakaveri/tanuh-buffer-tee/internal/jobs"
	"github.com/datakaveri/tanuh-buffer-tee/internal/provision"
)

// Scheduler wires the store, the provisioning engine, and the dispatcher.
type Scheduler struct {
	Cfg    config.Config
	Store  *jobs.Store
	Engine *provision.Engine
	Health provision.Health
	// Deliver is dispatch.Deliver, injectable for tests. writeBody streams
	// the request body; contentType carries its multipart boundary.
	Deliver func(ctx context.Context, addr, audience, digest, contentType string, writeBody func(io.Writer) error) error
}

// Run ticks until ctx is cancelled.
func (s *Scheduler) Run(ctx context.Context) {
	log.Printf("scheduler: started (interval=%s, dispatch_timeout=%s, retention=%s, max_requeue=%d)",
		s.Cfg.SchedulerTick, s.Cfg.DispatchTimeout, s.Cfg.JobRetention, s.Cfg.MaxRequeue)
	ticker := time.NewTicker(s.Cfg.SchedulerTick)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			log.Printf("scheduler: stopped")
			return
		case <-ticker.C:
			s.tick(ctx)
		}
	}
}

func (s *Scheduler) tick(ctx context.Context) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("scheduler: tick panic recovered: %v", r)
		}
	}()

	for _, id := range s.Store.CleanupOld(s.Cfg.JobRetention) {
		log.Printf("scheduler: deleted job %s (retention elapsed)", id)
	}
	for _, id := range s.Store.StalledQueued(s.Cfg.QueueStallAfter) {
		log.Printf("scheduler: WARNING job %s queued longer than %s with no dispatch", id, s.Cfg.QueueStallAfter)
	}

	s.recoverStuckDispatched(ctx)
	s.dispatchNext(ctx)
}

// recoverStuckDispatched requeues (bounded) dispatched jobs whose TEE died
// without delivering the completion callback.
func (s *Scheduler) recoverStuckDispatched(ctx context.Context) {
	dispatched, err := s.Store.Dispatched()
	if err != nil {
		log.Printf("scheduler: list dispatched: %v", err)
		return
	}
	now := time.Now().Unix()
	for _, job := range dispatched {
		if now-job.DispatchedAtUnix < int64(s.Cfg.DispatchTimeout.Seconds()) {
			continue
		}
		// A live eval keeps the TEE healthy — leave it alone regardless of
		// elapsed time.
		if job.AssignedTEE != "" && s.Health.Healthy(ctx, "https://"+job.AssignedTEE+"/healthz") {
			continue
		}
		if job.RequeueCount >= s.Cfg.MaxRequeue {
			log.Printf("scheduler: job %s stuck with no callback and requeue budget exhausted (%d) — marking error",
				job.JobID, job.RequeueCount)
			if _, err := s.Store.Fail(job.JobID, "requeue_exhausted", map[string]any{
				"error_type":    "DispatchLost",
				"error_message": fmt.Sprintf("no completion callback after %d dispatches", job.RequeueCount+1),
			}); err != nil {
				log.Printf("scheduler: fail job %s: %v", job.JobID, err)
			}
			continue
		}
		log.Printf("scheduler: job %s dispatched %ds ago with no callback and TEE unhealthy — requeueing (%d/%d)",
			job.JobID, now-job.DispatchedAtUnix, job.RequeueCount+1, s.Cfg.MaxRequeue)
		if _, err := s.Store.Requeue(job.JobID); err != nil {
			log.Printf("scheduler: requeue job %s: %v", job.JobID, err)
		}
	}
}

// dispatchNext provisions a target for the next queued job and delivers it.
// One job in flight at a time, matching the Processing TEE's 409 gate.
func (s *Scheduler) dispatchNext(ctx context.Context) {
	busy, err := s.Store.AnyDispatched()
	if err != nil || busy {
		return
	}
	job, err := s.Store.NextQueued()
	if err != nil || job == nil {
		return
	}

	log.Printf("scheduler: dispatch cycle: job %s (dataset %s)", job.JobID, job.DatasetID)
	s.Store.UpdateDispatchState(map[string]any{
		"last_dispatch_attempt_unix": time.Now().Unix(),
		"last_dispatch_job_id":       job.JobID,
		"last_dispatch_status":       "checking_vm_health",
		"last_dispatch_error":        "",
	})

	// Check the payload before provisioning: a job whose artifacts or upload
	// keys are gone can never run, so fail it instead of starting a VM and
	// retrying it every tick.
	p, err := s.preparePayload(job)
	if err != nil {
		log.Printf("scheduler: prepare payload for %s: %v — failing job", job.JobID, err)
		if _, ferr := s.Store.Fail(job.JobID, "buffer_dispatch", map[string]any{
			"error_type":    "ArtifactUnavailable",
			"error_message": err.Error(),
		}); ferr != nil {
			log.Printf("scheduler: fail job %s: %v", job.JobID, ferr)
		}
		return
	}
	defer p.clear()

	res, err := s.Engine.Provision(ctx, job)
	if err != nil {
		log.Printf("scheduler: provision error for %s: %v", job.JobID, err)
		return
	}
	if res == nil {
		log.Printf("scheduler: job %s: no Processing TEE available this cycle — leaving queued", job.JobID)
		return
	}

	if err := s.Deliver(ctx, res.Target.Addr, s.Cfg.DispatchAudience, res.Target.ImageDigest, p.contentType(), p.write); err != nil {
		s.dispatchFailed(job, err)
		return
	}

	if _, err := s.Store.MarkDispatched(job.JobID, res.Target.Addr); err != nil {
		log.Printf("scheduler: mark dispatched %s: %v", job.JobID, err)
		return
	}
	s.Store.UpdateDispatchState(map[string]any{
		"last_dispatch_attempt_unix": time.Now().Unix(),
		"last_dispatch_job_id":       job.JobID,
		"last_dispatch_status":       "dispatched",
		"last_dispatch_error":        "",
	})
	log.Printf("scheduler: job %s dispatched successfully via RA-TLS (%s)", job.JobID, res.Target.Name)
}

// dispatchFailed records a failed delivery. A refusal the Processing TEE will
// repeat for this payload (a bad upload, a version mismatch, no disk for it)
// fails the job, so it is not resent — and a VM started — every tick. Anything
// else (busy, network, a stall) leaves the job queued for the next tick.
func (s *Scheduler) dispatchFailed(job *jobs.Job, err error) {
	log.Printf("scheduler: dispatch failed for %s: %v", job.JobID, err)
	status := "dispatch_failed"
	var rejected *dispatch.RejectedError
	if errors.As(err, &rejected) && rejected.Permanent() {
		status = "dispatch_rejected"
		log.Printf("scheduler: job %s refused by the Processing TEE (%d) — failing job", job.JobID, rejected.Status)
		if _, ferr := s.Store.Fail(job.JobID, "processing_intake", map[string]any{
			"error_type":    "PayloadRejected",
			"error_message": rejected.Message,
		}); ferr != nil {
			log.Printf("scheduler: fail job %s: %v", job.JobID, ferr)
		}
	}
	s.Store.UpdateDispatchState(map[string]any{
		"last_dispatch_attempt_unix": time.Now().Unix(),
		"last_dispatch_job_id":       job.JobID,
		"last_dispatch_status":       status,
		"last_dispatch_error":        err.Error(),
	})
}

// PayloadVersion identifies the Buffer → Processing TEE payload layout. The
// Processing TEE accepts only this version, so both sides ship together.
const PayloadVersion = 3

// payload is one job's dispatch body (version 3), streamed as
// multipart/form-data:
//
//	part "metadata"  {"payload_version": 3, "job_id", "dataset_id", "model_format",
//	                  "submitted_at_unix", "submitted_by", "keycloak_token", "buffer_job_url",
//	                  "artifacts": {"<slot>": {"encoding", "sha256", "ciphertext_size", "aes_key"}}}
//	part "<slot>"    the artifact's encrypted stream, exactly as the browser uploaded it
//
// The artifacts stay encrypted on disk and are copied straight into the
// request, so memory does not grow with their size. The AES keys travel
// inside the RA-TLS channel; the Processing TEE decrypts each artifact and
// checks it against the SHA-256 committed at submit.
type payload struct {
	boundary string
	metadata []byte // holds the AES keys: cleared after delivery
	parts    []payloadPart
}

type payloadPart struct {
	slot, path string
	size       int64
}

// preparePayload checks that every artifact the job expects is stored with
// its upload key, and builds the metadata part. Call clear when done.
func (s *Scheduler) preparePayload(job *jobs.Job) (*payload, error) {
	p := &payload{boundary: multipart.NewWriter(io.Discard).Boundary()}
	artifacts := map[string]any{}
	for _, a := range jobs.Artifacts() {
		if !a.Expected(job) {
			continue
		}
		stored, err := s.Store.Artifact(job.JobID, a.Slot)
		if err != nil {
			return nil, err
		}
		artifacts[a.Slot] = map[string]any{
			"encoding":        chunked.Encoding,
			"sha256":          a.Commitment(job),
			"ciphertext_size": stored.Size,
			"aes_key":         base64.StdEncoding.EncodeToString(stored.Key),
		}
		clear(stored.Key)
		p.parts = append(p.parts, payloadPart{slot: a.Slot, path: stored.Path, size: stored.Size})
	}
	meta, err := json.Marshal(map[string]any{
		"payload_version":   PayloadVersion,
		"job_id":            job.JobID,
		"dataset_id":        job.DatasetID,
		"model_format":      job.ModelFormat,
		"submitted_at_unix": job.SubmittedAtUnix,
		"submitted_by":      job.SubmittedBy,
		"keycloak_token":    job.KeycloakToken,
		"buffer_job_url":    s.Cfg.CallbackBase + "/v1/jobs/" + job.JobID,
		"artifacts":         artifacts,
	})
	if err != nil {
		return nil, err
	}
	p.metadata = meta
	return p, nil
}

func (p *payload) contentType() string { return "multipart/form-data; boundary=" + p.boundary }

// write streams the body to w: the metadata, then each artifact file. A file
// whose size changed since preparePayload fails the write.
func (p *payload) write(w io.Writer) error {
	mw := multipart.NewWriter(w)
	if err := mw.SetBoundary(p.boundary); err != nil {
		return err
	}
	part := func(name, contentType string) (io.Writer, error) {
		h := textproto.MIMEHeader{}
		h.Set("Content-Disposition", fmt.Sprintf("form-data; name=%q", name))
		h.Set("Content-Type", contentType)
		return mw.CreatePart(h)
	}
	pw, err := part("metadata", "application/json")
	if err != nil {
		return err
	}
	if _, err := pw.Write(p.metadata); err != nil {
		return err
	}
	for _, a := range p.parts {
		pw, err := part(a.slot, "application/octet-stream")
		if err != nil {
			return err
		}
		if err := copyFile(pw, a.path, a.size); err != nil {
			return fmt.Errorf("send %s: %w", a.slot, err)
		}
	}
	return mw.Close()
}

func copyFile(w io.Writer, path string, size int64) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	n, err := io.Copy(w, f)
	if err != nil {
		return err
	}
	if n != size {
		return fmt.Errorf("file is %d bytes, expected %d", n, size)
	}
	return nil
}

// clear wipes the metadata, which holds the AES keys.
func (p *payload) clear() { clear(p.metadata) }
