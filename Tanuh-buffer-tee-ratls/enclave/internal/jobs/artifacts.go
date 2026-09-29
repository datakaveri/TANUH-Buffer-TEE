package jobs

import "errors"

// Model formats a submission can declare. The Processing TEE loads each one
// with its own runtime; the buffer only enforces which upload slots apply.
const (
	FormatONNX        = "onnx"
	FormatTorchScript = "torchscript"
	FormatHuggingFace = "huggingface"
)

// ValidFormat reports whether f is a supported model format.
func ValidFormat(f string) bool {
	switch f {
	case FormatONNX, FormatTorchScript, FormatHuggingFace:
		return true
	}
	return false
}

// Upload slots. An artifact is stored on disk under its slot name.
const (
	SlotModel   = "model"   // .onnx / TorchScript .pt / Hugging Face .zip
	SlotWeights = "weights" // ONNX external weights (.onnx.data), optional
	SlotAdaptor = "adaptor" // adaptor.py: raw model outputs -> predictions.csv
)

var (
	// ErrUnknownSlot is returned for an upload slot that does not exist.
	ErrUnknownSlot = errors.New("unknown artifact slot")
	// ErrSlotNotAllowed is returned for a slot the job's model format does not use.
	ErrSlotNotAllowed = errors.New("artifact slot not used by this model format")
)

// ArtifactSpec is one upload slot: which formats use it, whether a job can be
// queued without it, its size limit, and the submit-time SHA-256 it must match.
// This table is the single definition used by the store, the upload route and
// the dispatch payload.
type ArtifactSpec struct {
	Slot           string
	AlwaysRequired bool // required for every job; otherwise required only when its hash was committed
	MaxBytes       int64
	allowed        func(format string) bool
	commitment     func(*Job) string
}

var artifactSpecs = []ArtifactSpec{
	{
		Slot:           SlotModel,
		AlwaysRequired: true,
		MaxBytes:       4 << 30,
		allowed:        func(string) bool { return true },
		commitment:     func(j *Job) string { return j.ModelSHA256 },
	},
	{
		Slot:       SlotWeights,
		MaxBytes:   4 << 30,
		allowed:    func(f string) bool { return f == FormatONNX },
		commitment: func(j *Job) string { return j.WeightsSHA256 },
	},
	{
		Slot:           SlotAdaptor,
		AlwaysRequired: true,
		MaxBytes:       1 << 20,
		allowed:        func(string) bool { return true },
		commitment:     func(j *Job) string { return j.AdaptorSHA },
	},
}

// Artifacts returns every upload slot, in dispatch order.
func Artifacts() []ArtifactSpec { return artifactSpecs }

// ArtifactBySlot looks up an upload slot by name.
func ArtifactBySlot(slot string) (ArtifactSpec, bool) {
	for _, a := range artifactSpecs {
		if a.Slot == slot {
			return a, true
		}
	}
	return ArtifactSpec{}, false
}

// AllowedFor reports whether the slot applies to a model format.
func (a ArtifactSpec) AllowedFor(format string) bool { return a.allowed(format) }

// Commitment returns the SHA-256 the job committed to for this slot ("" if none).
func (a ArtifactSpec) Commitment(j *Job) string { return a.commitment(j) }

// Expected reports whether a job must receive this artifact before it queues.
func (a ArtifactSpec) Expected(j *Job) bool {
	return a.AllowedFor(j.ModelFormat) && (a.AlwaysRequired || a.Commitment(j) != "")
}
