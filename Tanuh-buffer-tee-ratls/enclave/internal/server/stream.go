package server

import (
	"bufio"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// readIdleTimeout is how long an upload may stall before the connection is
// dropped. There is no limit on the whole upload: a multi-GB model at a
// steady rate takes as long as it needs. A variable so tests can shorten it.
var readIdleTimeout = 5 * time.Minute

// idleDeadlineReader pushes the request's read deadline readIdleTimeout ahead
// while bytes keep arriving (at most once a second, and at least four times
// per timeout), replacing the server's fixed ReadTimeout for this request.
// The write deadline moves with it: over HTTP/2 the server's WriteTimeout
// ends the stream even while the handler is still reading.
type idleDeadlineReader struct {
	r    io.Reader
	rc   *http.ResponseController
	next time.Time
}

func (d *idleDeadlineReader) Read(p []byte) (int, error) {
	if now := time.Now(); now.After(d.next) {
		_ = d.rc.SetReadDeadline(now.Add(readIdleTimeout))
		_ = d.rc.SetWriteDeadline(now.Add(readIdleTimeout + time.Minute))
		d.next = now.Add(min(time.Second, readIdleTimeout/4))
	}
	return d.r.Read(p)
}

// memoryUsage reports this process's current and peak resident memory.
func memoryUsage() string {
	f, err := os.Open("/proc/self/status")
	if err != nil {
		return "memory unknown"
	}
	defer f.Close()
	var out []string
	s := bufio.NewScanner(f)
	for s.Scan() {
		if line := s.Text(); strings.HasPrefix(line, "VmRSS:") || strings.HasPrefix(line, "VmHWM:") {
			out = append(out, strings.Join(strings.Fields(line), " "))
		}
	}
	return strings.Join(out, ", ")
}
