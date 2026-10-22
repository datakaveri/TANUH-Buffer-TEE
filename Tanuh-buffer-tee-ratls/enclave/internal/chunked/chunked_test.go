package chunked

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"io"
	"os"
	"testing"
)

// golden is testdata/chunked_v1/vector.json, the same file the Processing
// TEE's decrypter is tested against.
func golden(t *testing.T) []byte {
	t.Helper()
	raw, err := os.ReadFile("testdata/chunked_v1/vector.json")
	if err != nil {
		t.Fatal(err)
	}
	var v struct {
		Stream string `json:"stream_b64"`
	}
	if err := json.Unmarshal(raw, &v); err != nil {
		t.Fatal(err)
	}
	stream, err := base64.StdEncoding.DecodeString(v.Stream)
	if err != nil {
		t.Fatal(err)
	}
	return stream
}

// frames builds a stream with chunks of the given ciphertext lengths. The
// Buffer never decrypts, so the chunk bytes need not be real ciphertext.
func frames(total int, lengths ...int) (hdr []byte, chunks [][]byte) {
	h, _ := json.Marshal(Header{FileID: "f", BaseIV: base64.StdEncoding.EncodeToString(make([]byte, 12)), TotalChunks: total})
	hdr = append(binary.BigEndian.AppendUint32(nil, uint32(len(h))), h...)
	for i, n := range lengths {
		chunks = append(chunks, append(binary.BigEndian.AppendUint32(nil, uint32(n)), bytes.Repeat([]byte{byte(i)}, n)...))
	}
	return hdr, chunks
}

func join(parts ...[]byte) []byte { return bytes.Join(parts, nil) }

// check runs ReadHeader + CopyChunks as the upload handler does, returning
// what was copied.
func check(stream []byte) ([]byte, int64, error) {
	r := bytes.NewReader(stream)
	hdr, raw, err := ReadHeader(r)
	if err != nil {
		return nil, 0, err
	}
	out := bytes.NewBuffer(append([]byte(nil), raw...))
	n, err := CopyChunks(r, hdr, out)
	return out.Bytes(), n, err
}

func TestGoldenCopiedVerbatim(t *testing.T) {
	stream := golden(t)
	out, n, err := check(stream)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(out, stream) || n != 40 {
		t.Fatalf("copied %d bytes (plaintext %d), want the %d-byte stream unchanged and 40", len(out), n, len(stream))
	}
}

func TestRejects(t *testing.T) {
	hdr, c := frames(3, 30, 30, 20)
	good := join(hdr, c[0], c[1], c[2])
	if _, _, err := check(good); err != nil {
		t.Fatalf("good stream rejected: %v", err)
	}
	badHeader := func(h Header) []byte {
		raw, _ := json.Marshal(h)
		return join(binary.BigEndian.AppendUint32(nil, uint32(len(raw))), raw, c[0])
	}
	iv := base64.StdEncoding.EncodeToString(make([]byte, 12))
	cases := map[string][]byte{
		"empty":                    nil,
		"header length 0":          {0, 0, 0, 0},
		"header over 8 KiB":        binary.BigEndian.AppendUint32(nil, 8193),
		"header not JSON":          join(binary.BigEndian.AppendUint32(nil, 3), []byte("abc")),
		"header truncated":         hdr[:len(hdr)-2],
		"total_chunks 0":           badHeader(Header{FileID: "f", BaseIV: iv, TotalChunks: 0}),
		"no file_id":               badHeader(Header{BaseIV: iv, TotalChunks: 1}),
		"bad base_iv":              badHeader(Header{FileID: "f", BaseIV: "AAAA", TotalChunks: 1}),
		"missing last chunk":       join(hdr, c[0], c[1]),
		"truncated mid-chunk":      good[:len(good)-1],
		"trailing bytes":           join(good, []byte{0}),
		"extra chunk":              join(good, c[2]),
		"chunk shorter than a tag": join(hdr, binary.BigEndian.AppendUint32(nil, 15)),
		"chunk over 64 MiB + tag":  join(hdr, binary.BigEndian.AppendUint32(nil, MaxChunkBytes+1)),
	}
	for name, stream := range cases {
		if _, _, err := check(stream); err == nil {
			t.Errorf("%s: accepted", name)
		} else {
			t.Logf("%s: %v", name, err)
		}
	}
}

// A 64 MiB chunk (the largest the Processing TEE accepts) passes through
// without being held in memory.
func TestMaxChunkStreams(t *testing.T) {
	hdr, _ := frames(1)
	r := io.MultiReader(bytes.NewReader(hdr), bytes.NewReader(binary.BigEndian.AppendUint32(nil, MaxChunkBytes)),
		io.LimitReader(zeros{}, MaxChunkBytes))
	h, _, err := ReadHeader(r)
	if err != nil {
		t.Fatal(err)
	}
	n, err := CopyChunks(r, h, io.Discard)
	if err != nil || n != 64<<20 {
		t.Fatalf("n=%d err=%v", n, err)
	}
}

type zeros struct{}

func (zeros) Read(p []byte) (int, error) { clear(p); return len(p), nil }
