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

// control(): gate = W_down . [a; b];  dst = cur*(1 - tanh(W_forget . gate)) + tanh(W_gain . gate)*msg
// gate_tmp is scratch for the [rank, n_tokens] gate
void ggml_cuda_op_loop_control(ggml_backend_cuda_context & ctx, const ggml_tensor * a, const ggml_tensor * b,
                               const ggml_tensor * w_down, const ggml_tensor * w_forget, const ggml_tensor * w_gain,
                               const ggml_tensor * cur, const ggml_tensor * msg, ggml_tensor * gate_tmp, ggml_tensor * dst);

// situ(): dst = beta*tanh(g/beta)*sigmoid(g) * lb*tanh(u/lb), g and u are strided 2D views
void ggml_cuda_op_situ(ggml_backend_cuda_context & ctx, const ggml_tensor * g, const ggml_tensor * u, ggml_tensor * dst,
                       float inv_beta, float beta, float inv_lb, float lb);

// convolve() tail: dst[c, i] = (cur[c, i] + sum_j x[i + j, c]*w[j, c]) * scale
void ggml_cuda_op_conv_res_scale(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * w,
                                 const ggml_tensor * cur, ggml_tensor * dst, float scale);
