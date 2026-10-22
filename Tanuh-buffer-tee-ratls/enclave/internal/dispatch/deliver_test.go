package dispatch

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// reply returns a handler that reads the whole body, records it, and replies.
func reply(status int, body string, got *[]byte) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		if got != nil {
			*got = data
		}
		w.WriteHeader(status)
		io.WriteString(w, body) //nolint:errcheck
	}
}

func writeString(s string) func(io.Writer) error {
	return func(w io.Writer) error { _, err := io.WriteString(w, s); return err }
}

func TestPostStreamsTheBody(t *testing.T) {
	var got []byte
	srv := httptest.NewServer(reply(http.StatusAccepted, `{"status":"accepted"}`, &got))
	defer srv.Close()
	body := strings.Repeat("payload ", 1<<16)
	if err := post(context.Background(), srv.Client(), srv.URL, "multipart/form-data; boundary=x", writeString(body)); err != nil {
		t.Fatal(err)
	}
	if string(got) != body {
		t.Fatalf("server received %d bytes, want %d", len(got), len(body))
	}
}

func TestPostClassifiesRefusals(t *testing.T) {
	cases := []struct {
		status    int
		body      string
		message   string
		permanent bool
	}{
		{http.StatusConflict, `{"status":"busy","message":"Already running job job-1"}`, "Already running job job-1", false},
		{http.StatusBadRequest, `{"status":"error","message":"model does not match its submit-time SHA-256"}`, "model does not match its submit-time SHA-256", true},
		{http.StatusUnsupportedMediaType, `{"status":"error","message":"payload must be multipart/form-data (payload v3)"}`, "payload must be multipart/form-data (payload v3)", true},
		{http.StatusInsufficientStorage, `{"status":"error","message":"artifacts need 5 bytes of disk, 1 free"}`, "artifacts need 5 bytes of disk, 1 free", true},
		{http.StatusInternalServerError, `{"status":"error","message":"intake failed"}`, "intake failed", false},
		{http.StatusBadGateway, `not json`, "not json", false},
	}
	for _, c := range cases {
		srv := httptest.NewServer(reply(c.status, c.body, nil))
		err := post(context.Background(), srv.Client(), srv.URL, "x", writeString("body"))
		srv.Close()
		var re *RejectedError
		if !errors.As(err, &re) || re.Status != c.status || re.Message != c.message || re.Permanent() != c.permanent {
			t.Errorf("%d: err = %v (permanent %v), want message %q, permanent %v", c.status, err, re != nil && re.Permanent(), c.message, c.permanent)
		}
	}
}

func TestPostCancelsAStalledTransfer(t *testing.T) {
	saved := IdleTimeout
	IdleTimeout = 200 * time.Millisecond
	defer func() { IdleTimeout = saved }()

	srv := httptest.NewServer(reply(http.StatusAccepted, "", nil))
	defer srv.Close()
	stuck := func(w io.Writer) error {
		if _, err := w.Write([]byte("first bytes")); err != nil {
			return err
		}
		select {} // a payload source that never produces another byte
	}
	started := time.Now()
	err := post(context.Background(), srv.Client(), srv.URL, "x", stuck)
	if err == nil || !errors.Is(err, errStalled) {
		t.Fatalf("err = %v, want a stall", err)
	}
	if elapsed := time.Since(started); elapsed > 5*time.Second {
		t.Fatalf("stall noticed after %s", elapsed)
	}
}

// A slow but steady body is never cut off, however long it takes in total.
func TestPostAllowsASlowSteadyTransfer(t *testing.T) {
	saved := IdleTimeout
	IdleTimeout = 300 * time.Millisecond
	defer func() { IdleTimeout = saved }()

	var got []byte
	srv := httptest.NewServer(reply(http.StatusAccepted, "", &got))
	defer srv.Close()
	slow := func(w io.Writer) error {
		for i := 0; i < 10; i++ { // ~1 s in total, more than 3× the idle timeout
			if _, err := w.Write([]byte("chunk")); err != nil {
				return err
			}
			time.Sleep(100 * time.Millisecond)
		}
		return nil
	}
	if err := post(context.Background(), srv.Client(), srv.URL, "x", slow); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, bytes.Repeat([]byte("chunk"), 10)) {
		t.Fatalf("received %q", got)
	}
}

func TestPostReportsABodyError(t *testing.T) {
	srv := httptest.NewServer(reply(http.StatusAccepted, "", nil))
	defer srv.Close()
	broken := func(w io.Writer) error {
		w.Write([]byte("part")) //nolint:errcheck
		return errors.New("send model: file is 9 bytes, expected 8")
	}
	if err := post(context.Background(), srv.Client(), srv.URL, "x", broken); err == nil {
		t.Fatal("a body that failed to write was reported as delivered")
	}
}
