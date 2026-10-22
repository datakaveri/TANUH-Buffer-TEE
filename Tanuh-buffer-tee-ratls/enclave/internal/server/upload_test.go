package server

import (
	"bytes"
	"crypto/ecdh"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/datakaveri/tanuh-buffer-tee/internal/bundle"
	"github.com/datakaveri/tanuh-buffer-tee/internal/config"
	hpkepkg "github.com/datakaveri/tanuh-buffer-tee/internal/hpke"
	"github.com/datakaveri/tanuh-buffer-tee/internal/jobs"
	"github.com/datakaveri/tanuh-buffer-tee/internal/keys"
)

// uploadEnv is a Buffer server over a temp store with one ONNX job committed
// to modelPlain and adaptorPlain, plus a browser HPKE key to upload with.
type uploadEnv struct {
	srv        *httptest.Server
	client     *http.Client
	store      *jobs.Store
	job        *jobs.Job
	enclave    *keys.EnclaveKeys
	browserPub []byte
}

var (
	modelPlain   = append([]byte{0x08}, bytes.Repeat([]byte("model"), 1000)...)
	adaptorPlain = []byte("print('adaptor')")
)

func sha256Hex(b []byte) string {
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}

func newUploadEnv(t *testing.T) *uploadEnv {
	t.Helper()
	return newUploadEnvWith(t, func(srv *httptest.Server) { srv.Start() })
}

func newUploadEnvWith(t *testing.T, start func(*httptest.Server)) *uploadEnv {
	t.Helper()
	k, err := keys.Generate()
	if err != nil {
		t.Fatal(err)
	}
	store, err := jobs.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	job, err := store.Create(jobs.NewJobRequest{ModelFormat: jobs.FormatONNX,
		ModelSHA256: sha256Hex(modelPlain), AdaptorSHA: sha256Hex(adaptorPlain)})
	if err != nil {
		t.Fatal(err)
	}
	browser, err := ecdh.X25519().GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewUnstartedServer(New(bundle.NewWithKeys("", k), AuthConfig{}, store, config.Config{}).Handler())
	start(srv)
	t.Cleanup(srv.Close)
	return &uploadEnv{srv: srv, client: srv.Client(), store: store, job: job, enclave: k, browserPub: browser.PublicKey().Bytes()}
}

// stream builds an upload the way the browser does: a header carrying the
// AES key HPKE-wrapped to the enclave, then chunk frames. The Buffer never
// decrypts, so the chunk bytes are random stand-ins for ciphertext.
func (e *uploadEnv) stream(t *testing.T, aesKey []byte, plain []byte, chunks int) []byte {
	t.Helper()
	info := hpkepkg.ContextInfo(e.enclave.HPKEPub.Bytes(), e.browserPub)
	sealed, err := hpkepkg.SealToBrowser(e.enclave.HPKEPub.Bytes(), info, nil, aesKey)
	if err != nil {
		t.Fatal(err)
	}
	hdr, _ := json.Marshal(map[string]any{
		"file_id": "f", "enc": base64.StdEncoding.EncodeToString(sealed.Enc),
		"wrapped_key":  base64.StdEncoding.EncodeToString(sealed.Ciphertext),
		"base_iv":      base64.StdEncoding.EncodeToString(make([]byte, 12)),
		"total_chunks": chunks, "plaintext_sha256": sha256Hex(plain),
	})
	out := binary.BigEndian.AppendUint32(nil, uint32(len(hdr)))
	out = append(out, hdr...)
	per := (len(plain) + chunks - 1) / chunks
	for i := 0; i < chunks; i++ {
		ct := make([]byte, min(per, len(plain)-i*per)+16)
		rand.Read(ct) //nolint:errcheck
		out = binary.BigEndian.AppendUint32(out, uint32(len(ct)))
		out = append(out, ct...)
	}
	return out
}

func (e *uploadEnv) put(t *testing.T, slot string, body io.Reader) (int, map[string]any) {
	t.Helper()
	req, _ := http.NewRequest(http.MethodPut, e.srv.URL+"/v1/upload/"+e.job.JobID+"/"+slot, body)
	req.Header.Set("X-RATLS-Browser-HPKE", base64.RawURLEncoding.EncodeToString(e.browserPub))
	resp, err := e.client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out map[string]any
	json.NewDecoder(resp.Body).Decode(&out) //nolint:errcheck
	out["proto"] = resp.Proto
	return resp.StatusCode, out
}

// pacedReader hands out data a few bytes per read with a pause before each,
// and one long pause (stall) once stallAt bytes have gone.
type pacedReader struct {
	data    []byte
	step    int
	pause   time.Duration
	stallAt int
	stall   time.Duration
}

func (p *pacedReader) Read(b []byte) (int, error) {
	if len(p.data) == 0 {
		return 0, io.EOF
	}
	time.Sleep(p.pause)
	if p.stall > 0 && p.stallAt <= 0 {
		time.Sleep(p.stall)
		p.stall = 0
	}
	n := copy(b[:min(len(b), p.step)], p.data)
	p.data, p.stallAt = p.data[n:], p.stallAt-n
	return n, nil
}

// Over HTTP/2 and TLS, as browsers reach the Buffer: an upload taking longer
// than the server's ReadTimeout and WriteTimeout succeeds while bytes keep
// arriving; one that stops sending for longer than readIdleTimeout is cut off.
func TestUploadDeadlinesOverHTTP2(t *testing.T) {
	saved := readIdleTimeout
	readIdleTimeout = 600 * time.Millisecond
	defer func() { readIdleTimeout = saved }()
	startTLS := func(srv *httptest.Server) {
		srv.EnableHTTP2 = true
		srv.Config.ReadTimeout = time.Second
		srv.Config.WriteTimeout = time.Second
		srv.StartTLS()
	}

	e := newUploadEnvWith(t, startTLS)
	stream := e.stream(t, make([]byte, 32), modelPlain, 3)
	steady := &pacedReader{data: stream, step: len(stream)/20 + 1, pause: 100 * time.Millisecond} // ~2 s in all
	started := time.Now()
	code, out := e.put(t, "model", steady)
	if code != http.StatusOK || out["proto"] != "HTTP/2.0" {
		t.Fatalf("steady upload over %s: %d %v", out["proto"], code, out)
	}
	t.Logf("steady upload took %s over %s", time.Since(started).Round(time.Millisecond), out["proto"])

	e = newUploadEnvWith(t, startTLS)
	stream = e.stream(t, make([]byte, 32), modelPlain, 3)
	stalled := &pacedReader{data: stream, step: 512, stallAt: len(stream) / 2, stall: 2 * time.Second}
	req, _ := http.NewRequest(http.MethodPut, e.srv.URL+"/v1/upload/"+e.job.JobID+"/model", stalled)
	req.Header.Set("X-RATLS-Browser-HPKE", base64.RawURLEncoding.EncodeToString(e.browserPub))
	if resp, err := e.client.Do(req); err == nil {
		resp.Body.Close()
		if resp.StatusCode == http.StatusOK {
			t.Fatal("an upload that stalled past the idle timeout was accepted")
		}
		t.Logf("stalled upload: %d", resp.StatusCode)
	} else {
		t.Logf("stalled upload: %v", err)
	}
	e.noPartFiles(t)
}

// noPartFiles fails if an upload left a temp file behind.
func (e *uploadEnv) noPartFiles(t *testing.T) {
	t.Helper()
	parts, _ := filepath.Glob(filepath.Join(e.job.ArtifactDir, "*.part"))
	if len(parts) > 0 {
		t.Errorf("temp files left: %v", parts)
	}
}

func TestUploadStoresCiphertextVerbatim(t *testing.T) {
	e := newUploadEnv(t)
	key := bytes.Repeat([]byte{0x42}, 32)
	stream := e.stream(t, key, modelPlain, 3)
	code, out := e.put(t, "model", bytes.NewReader(stream))
	if code != http.StatusOK || out["status"] != jobs.StatusPendingUpload || out["bytes"] != float64(len(modelPlain)) {
		t.Fatalf("%d %v", code, out)
	}
	a, err := e.store.Artifact(e.job.JobID, "model")
	if err != nil {
		t.Fatal(err)
	}
	if stored, _ := os.ReadFile(a.Path); !bytes.Equal(stored, stream) {
		t.Fatal("stored file differs from the uploaded stream")
	}
	if !bytes.Equal(a.Key, key) {
		t.Fatal("stored key differs from the browser's AES key")
	}
	e.noPartFiles(t)

	code, out = e.put(t, "adaptor", bytes.NewReader(e.stream(t, key, adaptorPlain, 1)))
	if code != http.StatusOK || out["status"] != jobs.StatusQueued {
		t.Fatalf("adaptor: %d %v", code, out)
	}
}

func TestUploadRejects(t *testing.T) {
	cases := map[string]struct {
		slot string
		body func(e *uploadEnv) []byte
		code int
	}{
		"truncated": {"model", func(e *uploadEnv) []byte {
			s := e.stream(t, make([]byte, 32), modelPlain, 3)
			return s[:len(s)-10]
		}, http.StatusBadRequest},
		"trailing bytes": {"model", func(e *uploadEnv) []byte {
			return append(e.stream(t, make([]byte, 32), modelPlain, 3), 0)
		}, http.StatusBadRequest},
		"header hash is not the commitment": {"model", func(e *uploadEnv) []byte {
			return e.stream(t, make([]byte, 32), []byte("another model"), 1)
		}, http.StatusBadRequest},
		"key wrapped to another enclave": {"model", func(e *uploadEnv) []byte {
			other, _ := keys.Generate()
			saved := e.enclave
			e.enclave = other
			defer func() { e.enclave = saved }()
			return e.stream(t, make([]byte, 32), modelPlain, 1)
		}, http.StatusBadRequest},
		"not a stream": {"model", func(*uploadEnv) []byte { return []byte("hello") }, http.StatusBadRequest},
		"unknown slot": {"preprocessing", func(*uploadEnv) []byte { return nil }, http.StatusNotFound},
		"weights not committed": {"weights", func(e *uploadEnv) []byte {
			return e.stream(t, make([]byte, 32), modelPlain, 1)
		}, http.StatusBadRequest},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			e := newUploadEnv(t)
			code, out := e.put(t, c.slot, bytes.NewReader(c.body(e)))
			if code != c.code {
				t.Fatalf("%d %v, want %d", code, out, c.code)
			}
			t.Log(out["message"])
			e.noPartFiles(t)
			if _, err := e.store.Artifact(e.job.JobID, "model"); err == nil {
				t.Error("a rejected upload was stored")
			}
			// The slot is free again: a good upload now succeeds.
			if code, out := e.put(t, "model", bytes.NewReader(e.stream(t, make([]byte, 32), modelPlain, 2))); code != http.StatusOK {
				t.Fatalf("retry after rejection: %d %v", code, out)
			}
		})
	}
}

// frameReader produces an upload stream of n chunks of chunkSize bytes on
// the fly, reusing one buffer, so the test itself allocates almost nothing.
type frameReader struct {
	head      []byte
	chunk     []byte // length prefix + ciphertext, reused
	left, pos int
	inHead    bool
}

func (f *frameReader) Read(p []byte) (int, error) {
	if f.inHead {
		n := copy(p, f.head[f.pos:])
		if f.pos += n; f.pos == len(f.head) {
			f.inHead, f.pos = false, 0
		}
		return n, nil
	}
	if f.left == 0 {
		return 0, io.EOF
	}
	n := copy(p, f.chunk[f.pos:])
	if f.pos += n; f.pos == len(f.chunk) {
		f.left, f.pos = f.left-1, 0
	}
	return n, nil
}

// A 256 MiB upload goes through with a small, size-independent amount of
// allocation: the stream is copied to disk, never held in memory.
func TestUploadMemoryDoesNotGrowWithSize(t *testing.T) {
	if testing.Short() {
		t.Skip("writes 256 MiB")
	}
	e := newUploadEnv(t)
	const chunkSize, chunks = 16 << 20, 16
	plainSize := chunkSize * chunks
	// Commit to the hash the header will carry (the Buffer only compares it).
	big := e.stream(t, make([]byte, 32), []byte("x"), 1)
	hdrLen := 4 + int(binary.BigEndian.Uint32(big))
	var h map[string]any
	json.Unmarshal(big[4:hdrLen], &h) //nolint:errcheck
	h["total_chunks"] = chunks
	h["plaintext_sha256"] = strings.Repeat("ab", 32)
	hdr, _ := json.Marshal(h)
	job, _ := e.store.Create(jobs.NewJobRequest{ModelFormat: jobs.FormatONNX, ModelSHA256: strings.Repeat("ab", 32), AdaptorSHA: sha256Hex(adaptorPlain)})
	e.job = job

	chunk := append(binary.BigEndian.AppendUint32(nil, chunkSize+16), make([]byte, chunkSize+16)...)
	r := &frameReader{head: append(binary.BigEndian.AppendUint32(nil, uint32(len(hdr))), hdr...), chunk: chunk, left: chunks, inHead: true}

	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	code, out := e.put(t, "model", r)
	runtime.ReadMemStats(&after)
	if code != http.StatusOK || out["bytes"] != float64(plainSize) {
		t.Fatalf("%d %v", code, out)
	}
	allocated := after.TotalAlloc - before.TotalAlloc
	t.Logf("%d MiB upload allocated %.1f MiB", plainSize>>20, float64(allocated)/(1<<20))
	if allocated > 32<<20 {
		t.Fatalf("allocated %d bytes for a %d-byte upload; the stream is being buffered", allocated, plainSize)
	}
}
