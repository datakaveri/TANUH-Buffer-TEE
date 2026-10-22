package bundle

import (
	"context"
	"crypto/ecdh"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/datakaveri/tanuh-buffer-tee/internal/attest"
	"github.com/datakaveri/tanuh-buffer-tee/internal/keys"
)

// Builder holds the current enclave keys and OIDC token, refreshed on rotation.
// All fields are protected by mu.
type Builder struct {
	audience  string
	mu        sync.RWMutex
	keys      *keys.EnclaveKeys
	oidcToken string
	issuedAt  int64
}

// NewBuilder generates fresh keys, fetches an OIDC token, and verifies
// that the eat_nonce claim in the token matches the generated keys.
func NewBuilder(ctx context.Context, audience string) (*Builder, error) {
	k, err := keys.Generate()
	if err != nil {
		return nil, fmt.Errorf("generate keys: %w", err)
	}

	token, err := attest.FetchOIDC(ctx, k, audience)
	if err != nil {
		return nil, fmt.Errorf("fetch OIDC: %w", err)
	}

	if err := verifyEatNonce(token, k); err != nil {
		return nil, fmt.Errorf("eat_nonce verification: %w", err)
	}

	return &Builder{
		audience:  audience,
		keys:      k,
		oidcToken: token,
		issuedAt:  time.Now().Unix(),
	}, nil
}

// NewWithKeys returns a Builder over existing keys with no attestation token,
// for tests of the handlers that only need the HPKE keys. The serving path
// always uses NewBuilder.
func NewWithKeys(audience string, k *keys.EnclaveKeys) *Builder {
	return &Builder{audience: audience, keys: k, issuedAt: time.Now().Unix()}
}

// Build signs the browser's liveness nonce and returns an attestation bundle.
func (b *Builder) Build(nonce []byte) (*AttestBundle, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	sig, err := b.keys.SignNonce(nonce)
	if err != nil {
		return nil, fmt.Errorf("sign nonce: %w", err)
	}

	return &AttestBundle{
		OIDCToken:        b.oidcToken,
		BindingPubkeyB64: base64.StdEncoding.EncodeToString(b.keys.BindingSPKI),
		HPKEPubkeyB64:    base64.StdEncoding.EncodeToString(b.keys.HPKEPubBytes()),
		NonceSigB64:      base64.StdEncoding.EncodeToString(sig),
		IssuedAt:         b.issuedAt,
	}, nil
}

// RefreshOIDC re-fetches the OIDC token using the current keys.
// Called on OIDCEvery cadence by the rotator.
func (b *Builder) RefreshOIDC(ctx context.Context) error {
	b.mu.RLock()
	k := b.keys
	b.mu.RUnlock()

	token, err := attest.FetchOIDC(ctx, k, b.audience)
	if err != nil {
		return fmt.Errorf("refresh OIDC: %w", err)
	}
	if err := verifyEatNonce(token, k); err != nil {
		return fmt.Errorf("eat_nonce verification after refresh: %w", err)
	}

	b.mu.Lock()
	b.oidcToken = token
	b.issuedAt = time.Now().Unix()
	b.mu.Unlock()
	return nil
}

// FullRotate generates new keys and a new OIDC token, then atomically swaps them.
// In-flight requests complete with the old keys; subsequent requests use the new ones.
func (b *Builder) FullRotate(ctx context.Context) error {
	k, err := keys.Generate()
	if err != nil {
		return fmt.Errorf("generate new keys: %w", err)
	}

	token, err := attest.FetchOIDC(ctx, k, b.audience)
	if err != nil {
		return fmt.Errorf("fetch OIDC for rotated keys: %w", err)
	}
	if err := verifyEatNonce(token, k); err != nil {
		return fmt.Errorf("eat_nonce verification after rotation: %w", err)
	}

	b.mu.Lock()
	b.keys = k
	b.oidcToken = token
	b.issuedAt = time.Now().Unix()
	b.mu.Unlock()
	return nil
}

// HPKEKeys returns the current HPKE keypair. Safe for concurrent use.
func (b *Builder) HPKEKeys() (*ecdh.PrivateKey, *ecdh.PublicKey) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.keys.HPKEPriv, b.keys.HPKEPub
}

// verifyEatNonce parses the JWT payload and asserts that eat_nonce[0] and [1]
// match the key fingerprints. This binding guarantee must not be skipped.
func verifyEatNonce(token string, k *keys.EnclaveKeys) error {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return fmt.Errorf("token is not a three-part JWT")
	}

	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return fmt.Errorf("decode JWT payload: %w", err)
	}

	var claims struct {
		EatNonce interface{} `json:"eat_nonce"`
	}
	if err := json.Unmarshal(payload, &claims); err != nil {
		return fmt.Errorf("unmarshal JWT claims: %w", err)
	}

	nonces, err := toStringSlice(claims.EatNonce)
	if err != nil {
		return fmt.Errorf("eat_nonce field: %w", err)
	}
	if len(nonces) < 2 {
		return fmt.Errorf("eat_nonce must have at least 2 elements, got %d", len(nonces))
	}

	wantFP := k.BindingFingerprintB64URL()
	wantHP := k.HPKEPubB64URL()

	if nonces[0] != wantFP {
		return fmt.Errorf("eat_nonce[0] mismatch: got %q, want %q", nonces[0], wantFP)
	}
	if nonces[1] != wantHP {
		return fmt.Errorf("eat_nonce[1] mismatch: got %q, want %q", nonces[1], wantHP)
	}
	return nil
}

// toStringSlice coerces eat_nonce to []string regardless of whether the
// JWT encodes it as an array or a single string (some CS versions vary).
func toStringSlice(v interface{}) ([]string, error) {
	switch t := v.(type) {
	case []interface{}:
		out := make([]string, len(t))
		for i, item := range t {
			s, ok := item.(string)
			if !ok {
				return nil, fmt.Errorf("element %d is not a string", i)
			}
			out[i] = s
		}
		return out, nil
	case string:
		return []string{t}, nil
	default:
		return nil, fmt.Errorf("unexpected type %T", v)
	}
}
