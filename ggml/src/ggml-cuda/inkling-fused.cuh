#include "common.cuh"

// Fused glue kernels for single-stream decode of op-heavy graphs (Inkling 2).
// Each replaces a chain of tiny elementwise kernels with one launch.

// clip(): x * min(1, thr / (||x * scale||_2 + eps)) per row of ne[0] elements, scale optional
void ggml_cuda_op_clip_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * scale,
                            ggml_tensor * dst, float thr, float eps, float post_scale);

#define HEPH_MIX_MAX_STATES 16
struct heph_mix_states {
    const float * p[HEPH_MIX_MAX_STATES];
};

// mix(): out[:, q, t] = sum_s softmax_s(<r_s, query_q> * sm_scale) * r_s, r_s = rms_norm(state_s[:, t])
void ggml_cuda_op_mix_states(ggml_backend_cuda_context & ctx, const heph_mix_states & states, int n_states,
                             const ggml_tensor * query, ggml_tensor * dst, float eps, float sm_scale);
