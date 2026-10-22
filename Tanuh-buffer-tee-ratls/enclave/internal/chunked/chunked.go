// Package chunked reads the AES-256-GCM-CHUNKED-v1 stream the browser uploads
// model files in:
//
//	[u32 BE header length][header JSON]
//	for each chunk i in 0..total_chunks-1:
//	  [u32 BE ciphertext length][ciphertext = plaintext + 16-byte GCM tag]
//
// The Buffer TEE stores the stream exactly as uploaded and forwards it to the
// Processing TEE, which decrypts and authenticates it (its
// internal/crypto/chunked.go; both are tested against testdata/chunked_v1).
// Here only the framing is checked — that the whole stream the browser framed
// has arrived — so nothing is decrypted and no chunk is held in memory.
package chunked

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

// Encoding names the format in the payload sent to the Processing TEE.
const Encoding = "AES-256-GCM-CHUNKED-v1"

const (
	// TagSize is the GCM tag at the end of every chunk.
	TagSize = 16
	// maxHeaderBytes bounds the header JSON, which is parsed whole.
	maxHeaderBytes = 8192
	// MaxChunkBytes is the largest chunk the Processing TEE accepts (64 MiB
	// of plaintext + tag): it must hold a whole chunk to authenticate it.
	// Rejecting larger chunks here fails such an upload at once rather than
	// when the job runs.
	MaxChunkBytes = 64<<20 + TagSize
)

// Header is frame 0. Enc and WrappedKey are the AES key HPKE-wrapped to the
// Buffer TEE; PlaintextSHA256 is optional.
type Header struct {
	FileID          string `json:"file_id"`
	Enc             string `json:"enc"`
	WrappedKey      string `json:"wrapped_key"`
	BaseIV          string `json:"base_iv"`
	TotalChunks     int    `json:"total_chunks"`
	PlaintextSHA256 string `json:"plaintext_sha256"`
}

// ReadHeader reads and checks frame 0, returning it parsed and as the raw
// bytes read (length prefix included), so the caller can store the stream
// verbatim.
func ReadHeader(r io.Reader) (Header, []byte, error) {
	var hdr Header
	frame := make([]byte, 4)
	if _, err := io.ReadFull(r, frame); err != nil {
		return hdr, nil, fmt.Errorf("read header length: %w", err)
	}
	hlen := binary.BigEndian.Uint32(frame)
	if hlen == 0 || hlen > maxHeaderBytes {
		return hdr, nil, fmt.Errorf("header length %d out of range", hlen)
	}
	frame = append(frame, make([]byte, hlen)...)
	if _, err := io.ReadFull(r, frame[4:]); err != nil {
		return hdr, nil, fmt.Errorf("read header: %w", err)
	}
	if err := json.Unmarshal(frame[4:], &hdr); err != nil {
		return hdr, nil, fmt.Errorf("parse header: %w", err)
	}
	if hdr.TotalChunks <= 0 {
		return hdr, nil, fmt.Errorf("total_chunks %d must be positive", hdr.TotalChunks)
	}
	if hdr.FileID == "" {
		return hdr, nil, fmt.Errorf("missing file_id")
	}
	if iv, err := base64.StdEncoding.DecodeString(hdr.BaseIV); err != nil || len(iv) != 12 {
		return hdr, nil, fmt.Errorf("base_iv must be 12 bytes of base64")
	}
	return hdr, frame, nil
}

// CopyChunks copies the header's total_chunks chunk frames from r to w
// unchanged, checking each chunk's length, that none is missing, and that r
// ends right after the last one. It returns the plaintext size the chunks
// hold. Nothing is decrypted: memory is one small copy buffer.
func CopyChunks(r io.Reader, hdr Header, w io.Writer) (plainSize int64, err error) {
	var lenBuf [4]byte
	for i := 0; i < hdr.TotalChunks; i++ {
		if _, err := io.ReadFull(r, lenBuf[:]); err != nil {
			return plainSize, fmt.Errorf("chunk %d/%d: read length: %w", i+1, hdr.TotalChunks, err)
		}
		clen := binary.BigEndian.Uint32(lenBuf[:])
		if clen < TagSize || clen > MaxChunkBytes {
			return plainSize, fmt.Errorf("chunk %d/%d: length %d out of range (16 to %d)", i+1, hdr.TotalChunks, clen, MaxChunkBytes)
		}
		if _, err := w.Write(lenBuf[:]); err != nil {
			return plainSize, err
		}
		if n, err := io.CopyN(w, r, int64(clen)); err != nil {
			if errors.Is(err, io.EOF) {
				return plainSize, fmt.Errorf("chunk %d/%d: truncated after %d of %d bytes", i+1, hdr.TotalChunks, n, clen)
			}
			return plainSize, fmt.Errorf("chunk %d/%d: %w", i+1, hdr.TotalChunks, err)
		}
		plainSize += int64(clen) - TagSize
	}
	var b [1]byte
	if n, err := io.ReadFull(r, b[:]); n > 0 {
		return plainSize, fmt.Errorf("unexpected bytes after the last chunk")
	} else if !errors.Is(err, io.EOF) {
		return plainSize, fmt.Errorf("read after the last chunk: %w", err)
	}
	return plainSize, nil
}
