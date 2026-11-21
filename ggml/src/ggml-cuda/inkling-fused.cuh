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

// control() including the two rms_norm inputs, reading cur and msg as [n_embd, n_tokens] views with a column stride
void ggml_cuda_op_loop_control_raw(ggml_backend_cuda_context & ctx, const ggml_tensor * cur, const ggml_tensor * msg,
                                   float eps_cur, float eps_msg,
                                   const ggml_tensor * w_down, const ggml_tensor * w_forget, const ggml_tensor * w_gain,
                                   ggml_tensor * gate_tmp, ggml_tensor * dst);

// MoE weight normalisation: dst[k, t] = w[k, t] / (sum_k w[k, t] * s + b) * out_scale
void ggml_cuda_op_moe_weights_norm(ggml_backend_cuda_context & ctx, const ggml_tensor * w, ggml_tensor * dst,
                                   float s, float b, float out_scale);

// relative-position attention bias joined with the mask, F16 out:
// dst[k, t, h] = f16((rel_pos[k, t] < n_extent ? <bias[:, rel_pos[k, t]], r[:, h, t]> : 0) - sinks[h, t] + mask[k, t])
void ggml_cuda_op_rel_bias_mask(ggml_backend_cuda_context & ctx, const ggml_tensor * bias, const void * r,
                                const void * sinks, const ggml_tensor * rel_pos, const ggml_tensor * mask, ggml_tensor * dst,
                                void * scratch, ggml_tensor * fill, float fill_v);

// convolve() head: concat of the recurrent past with the new columns, plus the state store
void ggml_cuda_op_conv_concat_store(ggml_backend_cuda_context & ctx, const ggml_tensor * past, const ggml_tensor * cur,
                                    ggml_tensor * cat, void * state, int n_t);

// whole convolve() without dynamic taps (n_conv == 4): state store, convolution, residual and scale in one kernel
// res: optional residual stream added to the result (the layer's residual add that follows convolve())
// ws_x, ws_w: optional, single token only: cur is not read, it is sum_k ws_w[k]*ws_x[:, k] (the MoE weighted sum, <= 8 experts)
void ggml_cuda_op_conv_full(ggml_backend_cuda_context & ctx, const ggml_tensor * past, const ggml_tensor * cur, const ggml_tensor * w,
                            const ggml_tensor * res, ggml_tensor * dst, void * state, float scale,
                            const ggml_tensor * ws_x = nullptr, const ggml_tensor * ws_w = nullptr);

#define HEPH_ROT_PRE_NONE 0
#define HEPH_ROT_PRE_CLIP 1
#define HEPH_ROT_PRE_RMS  2
struct heph_rot_args {
    const float * x;          // [128, rows] input, or the residual cur of the convolution when x4 is set
    const float * scale;      // optional per-element scale, scale_rows rows, cycled
    const float * x4;         // optional convolve() concat buffer [x_len, n_ch]
    const float * w;          // convolution taps [4, n_ch]
    const float * rot;        // [128, 128]
    float *       dst;
    int           scale_rows;
    int           n_ch;
    int           x_len;
    int           pre;
    float         conv_scale;
    float         thr;
    float         eps;
};
// [convolve() tail] -> [clip | rms_norm] -> rotation, rows of 128 elements, 4 convolution taps
void ggml_cuda_op_rot_rows(ggml_backend_cuda_context & ctx, const heph_rot_args & a, int64_t n_rows);

// q8_1 copy of a single F32 vector in the backend's small cache, or nullptr if the tensor does not qualify.
// quantize: launch the quantize kernel on a miss (otherwise the caller writes the slot). hit: optional, set on a cache hit.
const char * ggml_cuda_heph_q8_1(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, bool quantize, bool * hit = nullptr);

// several Q8_0 matrix x vector products (MUL_MAT nodes) in one launch
void ggml_cuda_op_mmv_q8_0_multi(ggml_backend_cuda_context & ctx, ggml_tensor ** nodes, int n);

// rms_norm(x)*w for one vector, F32 result plus its q8_1 form written to q8_1
void ggml_cuda_op_rms_norm_mul_q8(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * w, ggml_tensor * dst,
                                  void * q8_1, float eps);

// Row jobs: up to four independent row-wise chains in one launch, one block of 128 threads per row.
//   source: rows of `width` elements from x (times an optional scale), or the convolve() tail
//           (cur + sum_j x4[t + j, c]*(w[j, c] + dyn[n_conv - 1 - j, c/128, t]))*conv_scale with x4 = [past | cur],
//           which also stores the next conv state; dyn are the optional input-dependent taps
//   pre:    none, clip to an L2 norm of thr, or rms_norm
//   rot:    optional dense 128x128 product (width 128 only)
//   store:  optionally the rotated row is also packed into a turbo3 KV cache block, as SET_ROWS would
#define HEPH_ROW_MAX_JOBS 4
struct heph_row_job {
    const float * x;           // row source, or the conv residual cur [n_ch, n_t]
    const float * scale;       // optional, scale_rows rows of width elements, cycled
    const float * past;        // conv: [3, n_ch] with a column stride of past_stride elements
    const float * w;           // conv taps [4, n_ch]
    const float * rot;         // optional [128, 128]
    float *       dst;
    float *       state;       // conv: next state, [3, n_ch] dense
    const float * gate;        // optional: the result is multiplied by sigmoid(gate), same shape (attention output gate)
    void *        q8;          // optional: q8_1 form of the F32 result, one block per warp (rows of 128)
    const float * dyn;         // conv: optional input-dependent taps [n_conv*heads, n_t], added to w per head (128 channels)
    const float * post;        // optional factor per token applied after pre (log-scaled position), rows_per_token rows each
    int           dyn_stride;  // taps per token
    int           rows_per_token;
    void *        t3;          // optional turbo3 cache to store the rotated rows in (SET_ROWS), one block per row
    const int64_t * t3_idx;    // cache row of each token
    int64_t       t3_row_bytes;
    int           t3_heads;    // rows (heads) per token
    int           skip_f32;    // the F32 result has no reader besides the cache store
    int64_t       past_stride;
    uint32_t      block_end;   // one past the last row block of the job
    int           scale_rows;
    int           n_ch;
    int           n_t;
    int           width;
    int           pre;         // HEPH_ROT_PRE_*
    float         conv_scale;
    float         thr;
    float         eps;
};
struct heph_row_jobs {
    heph_row_job job[HEPH_ROW_MAX_JOBS];
    int          n;
};
void ggml_cuda_op_row_jobs(ggml_backend_cuda_context & ctx, const heph_row_jobs & jobs);

// control() for up to four independent streams at once (the Q, K and V inputs of a looped layer):
// one launch for all gates, one for all applies. Each job as ggml_cuda_op_loop_control_raw.
#define HEPH_CTRL_MAX_JOBS 4
struct heph_ctrl_jobs {
    const float * cur[HEPH_CTRL_MAX_JOBS];
    const float * msg[HEPH_CTRL_MAX_JOBS];
    const float * w_down[HEPH_CTRL_MAX_JOBS];
    const float * w_forget[HEPH_CTRL_MAX_JOBS];
    const float * w_gain[HEPH_CTRL_MAX_JOBS];
    float *       gate[HEPH_CTRL_MAX_JOBS];   // scratch, [rank, n_tokens]
    float *       dst[HEPH_CTRL_MAX_JOBS];
    int64_t       cur_stride[HEPH_CTRL_MAX_JOBS];
    int64_t       msg_stride[HEPH_CTRL_MAX_JOBS];
    float         eps_cur[HEPH_CTRL_MAX_JOBS];
    float         eps_msg[HEPH_CTRL_MAX_JOBS];
    int           n;
};
void ggml_cuda_op_loop_control_jobs(ggml_backend_cuda_context & ctx, const heph_ctrl_jobs & jobs, int n_embd, int n_tokens, int rank);

// rms_norm(x)*w with F32 and q8_1 results for up to four vectors of equal length in one launch
#define HEPH_NORMQ_MAX_JOBS 4
struct heph_normq_jobs {
    const float * x[HEPH_NORMQ_MAX_JOBS];
    const float * w[HEPH_NORMQ_MAX_JOBS];
    float *       dst[HEPH_NORMQ_MAX_JOBS];
    void *        q8_1[HEPH_NORMQ_MAX_JOBS];
    float         eps[HEPH_NORMQ_MAX_JOBS];
    int           n;
};
void ggml_cuda_op_rms_norm_mul_q8_jobs(ggml_backend_cuda_context & ctx, const heph_normq_jobs & jobs, int n);

// MoE router for sigmoid gating with a selection bias: the n_used experts with the largest sigmoid(logit) + bias, in
// descending order, and their normalised weights w/(sum(w)*s + b)*out_scale with w = sigmoid(logit).
// logits [n_expert <= 128, n_tokens], ids written as int32 with ids_stride per token, weights [n_used, n_tokens].
void ggml_cuda_op_moe_route(ggml_backend_cuda_context & ctx, const ggml_tensor * logits, const ggml_tensor * bias,
                            void * ids, int64_t ids_stride, ggml_tensor * weights, int n_used, float s, float b, float out_scale);
