package dispatch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"
)

// IdleTimeout cancels a dispatch when no payload bytes have been sent (or,
// once all are sent, no reply has come) for this long. It is not a limit on
// the whole transfer: a multi-GB payload at a steady rate takes as long as it
// needs. A variable so tests can shorten it.
var IdleTimeout = 5 * time.Minute

var errStalled = errors.New("no progress sending the payload")

// RejectedError is a reply from the Processing TEE other than 200/202.
type RejectedError struct {
	Status  int
	Message string
}

func (e *RejectedError) Error() string {
	return fmt.Sprintf("processing TEE refused the payload (%d): %s", e.Status, e.Message)
}

// Permanent reports whether sending the same payload again cannot succeed: a
// 4xx other than busy (409) or a timeout, or a TEE without the disk for it
// (507). Anything else may succeed on a later attempt.
func (e *RejectedError) Permanent() bool {
	switch e.Status {
	case http.StatusConflict, http.StatusRequestTimeout, http.StatusTooManyRequests:
		return false
	case http.StatusInsufficientStorage:
		return true
	}
	return e.Status >= 400 && e.Status < 500
}

// Deliver performs one in-process job dispatch: RA-TLS connect to the
// Processing TEE at addr, verify its attestation against expectedDigest,
// and POST the secure payload to /api/load-model. The body is streamed:
// writeBody writes it into a pipe as the connection sends it, so it is never
// held in memory whatever the artifacts' size. contentType is the body's
// Content-Type (multipart, with its boundary). A refusal is a *RejectedError.
func Deliver(ctx context.Context, addr, audience, expectedDigest, contentType string, writeBody func(io.Writer) error) error {
	if expectedDigest == "" {
		return fmt.Errorf("dispatch: expected image digest must be set before dispatching")
	}

	log.Printf("dispatch: connecting to Processing TEE at %s", addr)
	session, err := Connect(ctx, addr, &VerificationOptions{
		Audience:            audience,
		ExpectedImageDigest: expectedDigest,
	})
	if err != nil {
		return fmt.Errorf("dispatch: RA-TLS failed: %w", err)
	}
	log.Printf("dispatch: RA-TLS verified — hwmodel=%s digest=%s", session.HWModel, session.ContainerImageDigest)

	started := time.Now()
	if err := post(ctx, session.Client, "https://"+addr+"/api/load-model", contentType, writeBody); err != nil {
		return err
	}
	log.Printf("dispatch: payload delivered successfully to %s in %s", addr, time.Since(started).Round(time.Millisecond))
	return nil
}

// post streams the body to url and checks the reply, cancelling the request
// if it stalls for IdleTimeout.
func post(ctx context.Context, client *http.Client, url, contentType string, writeBody func(io.Writer) error) error {
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)

	// The writer goroutine ends when the body is fully written, or when the
	// pipe's read end is closed (error, cancel or early reply). On cancel the
	// pipe is closed here too: the transport waits for its body copy to end,
	// which would otherwise block on a source that has stopped producing.
	pr, pw := io.Pipe()
	defer pr.Close()
	stop := context.AfterFunc(ctx, func() { pr.CloseWithError(context.Cause(ctx)) })
	defer stop()
	progress := make(chan struct{}, 1)
	go func() { pw.CloseWithError(writeBody(&progressWriter{w: pw, progress: progress})) }()
	go watchdog(ctx, cancel, progress)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, pr)
	if err != nil {
		return fmt.Errorf("dispatch: build POST /api/load-model: %w", err)
	}
	req.Header.Set("Content-Type", contentType)

	resp, err := client.Do(req)
	if err != nil {
		if errors.Is(context.Cause(ctx), errStalled) {
			return fmt.Errorf("dispatch: POST /api/load-model: %w for %s", errStalled, IdleTimeout)
		}
		return fmt.Errorf("dispatch: POST /api/load-model: %w", err)
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	log.Printf("dispatch: Processing TEE response status=%d body=%s", resp.StatusCode, string(respBody))
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusAccepted {
		var reply struct {
			Message string `json:"message"`
		}
		if json.Unmarshal(respBody, &reply) != nil || reply.Message == "" {
			reply.Message = string(respBody)
		}
		return &RejectedError{Status: resp.StatusCode, Message: reply.Message}
	}
	return nil
}

// progressWriter signals each successful write: the pipe only returns once
// the transport has taken the bytes, so it tracks what is really being sent.
type progressWriter struct {
	w        io.Writer
	progress chan<- struct{}
}

func (p *progressWriter) Write(b []byte) (int, error) {
	n, err := p.w.Write(b)
	if n > 0 {
		select {
		case p.progress <- struct{}{}:
		default:
		}
	}
	return n, err
}

// watchdog cancels the request when IdleTimeout passes with no progress.
func watchdog(ctx context.Context, cancel context.CancelCauseFunc, progress <-chan struct{}) {
	timer := time.NewTimer(IdleTimeout)
	defer timer.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-progress:
			timer.Reset(IdleTimeout)
		case <-timer.C:
			cancel(errStalled)
			return
		}
	}
}
