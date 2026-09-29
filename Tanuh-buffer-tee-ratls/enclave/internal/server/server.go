package server

import (
	"encoding/json"
	"log"
	"net/http"
	"os"

	"github.com/datakaveri/tanuh-buffer-tee/internal/bundle"
	"github.com/datakaveri/tanuh-buffer-tee/internal/config"
	"github.com/datakaveri/tanuh-buffer-tee/internal/jobs"
)

// Server wires HTTP handlers to a ServeMux. All job state lives in the
// in-process store — the former Flask :4100 proxy hop is gone.
type Server struct {
	builder *bundle.Builder
	store   *jobs.Store
	cfg     config.Config
	mux     *http.ServeMux
	auth    AuthConfig
}

// New creates a Server and registers all routes.
func New(b *bundle.Builder, auth AuthConfig, store *jobs.Store, cfg config.Config) *Server {
	s := &Server{builder: b, store: store, cfg: cfg, mux: http.NewServeMux(), auth: auth}
	s.mux.HandleFunc("GET /v1/attest", s.HandleAttest)
	s.mux.HandleFunc("POST /v1/submit", s.HandleSubmit)
	// Binary file upload — large files sent over TLS, hashes committed via HPKE.
	// {slot} is model | weights | adaptor (jobs.Artifacts); anything else is 404.
	s.mux.HandleFunc("PUT /v1/upload/{job_id}/{slot}", s.HandleUpload)
	s.mux.HandleFunc("GET /v1/status/{job_id}", s.HandleStatus)
	s.mux.Handle("GET /v1/queue", requireRole("org_admin", http.HandlerFunc(s.HandleQueue)))
	s.mux.HandleFunc("GET /v1/results/{job_id}", s.HandleResults)
	// Completion callback from the Processing TEE — authenticated by the
	// caller's CS attestation token (see complete.go), exempt from Keycloak.
	s.mux.HandleFunc("POST /v1/jobs/{job_id}/complete", s.HandleComplete)
	s.mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	return s
}

// Handler returns the HTTP handler with CORS and optional auth middleware.
func (s *Server) Handler() http.Handler {
	base := corsMiddleware(s.mux)
	if s.auth.JWKSUrl == "" {
		return base
	}
	return keycloakAuthMiddleware(s.auth.JWKSUrl, s.auth.Issuer, base)
}

// corsMiddleware adds CORS headers and handles OPTIONS preflights.
func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "X-RATLS-Nonce, X-RATLS-Browser-HPKE, Content-Type, Authorization")
		w.Header().Set("Access-Control-Max-Age", "86400")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// HandleStatus serves GET /v1/status/{job_id} from the store.
func (s *Server) HandleStatus(w http.ResponseWriter, r *http.Request) {
	job, err := s.store.Get(r.PathValue("job_id"))
	if err != nil {
		writeJSON(w, http.StatusNotFound, map[string]any{"status": "error", "message": "Unknown job_id"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "success", "job": job})
}

// HandleQueue serves GET /v1/queue from the store.
func (s *Server) HandleQueue(w http.ResponseWriter, _ *http.Request) {
	all, queued, err := s.store.List()
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"status": "error", "message": "store unavailable"})
		return
	}
	if queued == nil {
		queued = []string{}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"status":         "success",
		"queued_count":   len(queued),
		"queued_job_ids": queued,
		"jobs":           all,
	})
}

// HandleResults serves GET /v1/results/{job_id}. Results are delivered to
// the external leaderboard; this endpoint reports job status until (unless)
// a local results file exists.
func (s *Server) HandleResults(w http.ResponseWriter, r *http.Request) {
	jobID := r.PathValue("job_id")
	job, err := s.store.Get(jobID)
	if err != nil {
		writeJSON(w, http.StatusNotFound, map[string]any{"status": "error", "message": "Unknown job_id"})
		return
	}
	if raw, err := os.ReadFile(s.store.ResultsPath(jobID)); err == nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(raw) //nolint:errcheck
		return
	}
	writeJSON(w, http.StatusNotFound, map[string]any{
		"status":  job.Status,
		"message": "results not yet available",
	})
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("server: encode response: %v", err)
	}
}
