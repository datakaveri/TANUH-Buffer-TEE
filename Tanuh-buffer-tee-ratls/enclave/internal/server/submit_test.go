package server

import (
	"strings"
	"testing"
)

var hex64 = strings.Repeat("ab", 32)

func TestJobRequestValidate(t *testing.T) {
	cases := []struct {
		name    string
		req     JobRequest
		wantErr string
	}{
		{"onnx with weights", JobRequest{ModelFormat: "onnx", ModelSHA256: hex64, WeightsSHA256: hex64, AdaptorSHA256: hex64}, ""},
		{"onnx without weights", JobRequest{ModelFormat: "onnx", ModelSHA256: hex64, AdaptorSHA256: hex64}, ""},
		{"torchscript", JobRequest{ModelFormat: "torchscript", ModelSHA256: hex64, AdaptorSHA256: hex64}, ""},
		{"huggingface upper-case hex", JobRequest{ModelFormat: "huggingface", ModelSHA256: strings.ToUpper(hex64), AdaptorSHA256: hex64}, ""},
		{"missing format", JobRequest{ModelSHA256: hex64, AdaptorSHA256: hex64}, "model_format"},
		{"unknown format", JobRequest{ModelFormat: "keras", ModelSHA256: hex64, AdaptorSHA256: hex64}, "model_format"},
		{"missing adaptor hash", JobRequest{ModelFormat: "onnx", ModelSHA256: hex64}, "adaptor_sha256"},
		{"short model hash", JobRequest{ModelFormat: "onnx", ModelSHA256: "abcd", AdaptorSHA256: hex64}, "model_sha256"},
		{"non-hex model hash", JobRequest{ModelFormat: "onnx", ModelSHA256: strings.Repeat("zz", 32), AdaptorSHA256: hex64}, "model_sha256"},
		{"weights on torchscript", JobRequest{ModelFormat: "torchscript", ModelSHA256: hex64, WeightsSHA256: hex64, AdaptorSHA256: hex64}, "only used with model_format onnx"},
		{"malformed weights hash", JobRequest{ModelFormat: "onnx", ModelSHA256: hex64, WeightsSHA256: "x", AdaptorSHA256: hex64}, "weights_sha256"},
		{"torchscript with input spec", JobRequest{ModelFormat: "torchscript", ModelSHA256: hex64, AdaptorSHA256: hex64, InputSpecSHA256: hex64}, ""},
		{"onnx with input spec", JobRequest{ModelFormat: "onnx", ModelSHA256: hex64, AdaptorSHA256: hex64, InputSpecSHA256: hex64}, ""},
		{"input spec on huggingface", JobRequest{ModelFormat: "huggingface", ModelSHA256: hex64, AdaptorSHA256: hex64, InputSpecSHA256: hex64}, "only used with onnx and torchscript"},
		{"malformed input spec hash", JobRequest{ModelFormat: "onnx", ModelSHA256: hex64, AdaptorSHA256: hex64, InputSpecSHA256: "x"}, "input_spec_sha256"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := tc.req
			err := r.validate()
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if r.ModelSHA256 != strings.ToLower(r.ModelSHA256) {
					t.Fatal("hash not lower-cased")
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("err = %v, want mention of %q", err, tc.wantErr)
			}
		})
	}
}
