#include "inkling-fused.cuh"

template <int block_size>
static __global__ void clip_rows_f32(const float * x, const float * scale, float * dst,
                                     const int ncols, const int scale_rows,
                                     const float thr, const float eps, const float post_scale) {
    ggml_cuda_pdl_lc();
    const int64_t row = blockIdx.x;
    const int     tid = threadIdx.x;

    const float * xr = x   + row*ncols;
    float       * dr = dst + row*ncols;
    const float * sr = scale ? scale + (row % scale_rows)*ncols : nullptr;

    ggml_cuda_pdl_sync();

    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float v = sr ? xr[col]*sr[col] : xr[col];
        tmp += v*v;
    }

    tmp = warp_reduce_sum(tmp);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size <= 1024, "unexpected block_size");
        __shared__ float s_sum[32];
        const int warp_id = tid / WARP_SIZE;
        const int lane_id = tid % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = lane_id < block_size / WARP_SIZE ? s_sum[lane_id] : 0.0f;
        tmp = warp_reduce_sum(tmp);
    }

    const float norm   = sqrtf(tmp) + eps;
    const float factor = fminf(fmaxf(thr / norm, 0.0f), 1.0f) * post_scale;

    for (int col = tid; col < ncols; col += block_size) {
        const float v = sr ? xr[col]*sr[col] : xr[col];
        dr[col] = v*factor;
    }
}

void ggml_cuda_op_clip_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * scale,
                            ggml_tensor * dst, float thr, float eps, float post_scale) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(dst));
    GGML_ASSERT(scale == nullptr || (scale->type == GGML_TYPE_F32 && ggml_is_contiguous(scale) && scale->ne[0] == x->ne[0]));

    const int     ncols      = x->ne[0];
    const int64_t nrows      = ggml_nelements(x) / ncols;
    const int     scale_rows = scale ? (int) (ggml_nelements(scale) / ncols) : 1;

    const float * scale_d = scale ? (const float *) scale->data : nullptr;
    cudaStream_t  stream  = ctx.stream();

    if (ncols <= 64) {
        const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3(nrows, 1, 1), dim3(32, 1, 1), 0, stream);
        ggml_cuda_kernel_launch(clip_rows_f32<32>, lp, (const float *) x->data, scale_d, (float *) dst->data, ncols, scale_rows, thr, eps, post_scale);
    } else {
        const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3(nrows, 1, 1), dim3(128, 1, 1), 0, stream);
        ggml_cuda_kernel_launch(clip_rows_f32<128>, lp, (const float *) x->data, scale_d, (float *) dst->data, ncols, scale_rows, thr, eps, post_scale);
    }
}

#define HEPH_MIX_BLOCK 256

static __global__ void mix_states_f32(const heph_mix_states states, const int n_states, const float * query, float * dst,
                                      const int n_embd, const int nq, const float eps, const float sm_scale) {
    ggml_cuda_pdl_lc();
    const int q   = blockIdx.x;
    const int t   = blockIdx.y;
    const int tid = threadIdx.x;

    constexpr int n_warps = HEPH_MIX_BLOCK / WARP_SIZE;
    __shared__ float s_ss[HEPH_MIX_MAX_STATES][n_warps];
    __shared__ float s_dt[HEPH_MIX_MAX_STATES][n_warps];

    const float * qv = query + (int64_t) q*n_embd;

    ggml_cuda_pdl_sync();

    for (int s = 0; s < n_states; ++s) {
        const float * x = states.p[s] + (int64_t) t*n_embd;
        float ss = 0.0f;
        float dt = 0.0f;
        for (int e = tid; e < n_embd; e += HEPH_MIX_BLOCK) {
            const float v = x[e];
            ss += v*v;
            dt += v*qv[e];
        }
        ss = warp_reduce_sum(ss);
        dt = warp_reduce_sum(dt);
        if (tid % WARP_SIZE == 0) {
            s_ss[s][tid / WARP_SIZE] = ss;
            s_dt[s][tid / WARP_SIZE] = dt;
        }
    }
    __syncthreads();

    // every thread finishes the reduction and the softmax itself, n_states is tiny
    float w[HEPH_MIX_MAX_STATES]; // rms scale of each state, then its final weight
    float l[HEPH_MIX_MAX_STATES]; // softmax logits
    float max_val = -INFINITY;
    for (int s = 0; s < n_states; ++s) {
        float ss = 0.0f;
        float dt = 0.0f;
        for (int k = 0; k < n_warps; ++k) {
            ss += s_ss[s][k];
            dt += s_dt[s][k];
        }
        w[s]    = 1.0f / sqrtf(ss / n_embd + eps);
        l[s]    = dt*w[s]*sm_scale;
        max_val = fmaxf(max_val, l[s]);
    }
    float sum = 0.0f;
    for (int s = 0; s < n_states; ++s) {
        l[s] = expf(l[s] - max_val);
        sum += l[s];
    }
    for (int s = 0; s < n_states; ++s) {
        w[s] *= l[s] / sum;
    }

    float * out = dst + ((int64_t) t*nq + q)*n_embd;
    for (int e = tid; e < n_embd; e += HEPH_MIX_BLOCK) {
        float acc = 0.0f;
        for (int s = 0; s < n_states; ++s) {
            acc += w[s]*states.p[s][(int64_t) t*n_embd + e];
        }
        out[e] = acc;
    }
}

void ggml_cuda_op_mix_states(ggml_backend_cuda_context & ctx, const heph_mix_states & states, int n_states,
                             const ggml_tensor * query, ggml_tensor * dst, float eps, float sm_scale) {
    GGML_ASSERT(n_states >= 1 && n_states <= HEPH_MIX_MAX_STATES);
    GGML_ASSERT(query->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && ggml_is_contiguous(query) && ggml_is_contiguous(dst));

    const int n_embd = dst->ne[0];
    const int nq     = dst->ne[1];
    const int nt     = dst->ne[2];
    GGML_ASSERT(query->ne[0] == n_embd && query->ne[1] == nq);

    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3(nq, nt, 1), dim3(HEPH_MIX_BLOCK, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(mix_states_f32, lp, states, n_states, (const float *) query->data, (float *) dst->data, n_embd, nq, eps, sm_scale);
}

#define HEPH_CTRL_BLOCK 256

// one block per (gate row, token): gate[g, t] = sum_e w_down[e, g] * [a; b][e, t]
static __global__ void loop_control_gate_f32(const float * a, const float * b, const float * w_down, float * gate,
                                             const int n_embd, const int rank) {
    ggml_cuda_pdl_lc();
    const int g   = blockIdx.x;
    const int t   = blockIdx.y;
    const int tid = threadIdx.x;

    const float * wr = w_down + (int64_t) g*2*n_embd;
    const float * at = a + (int64_t) t*n_embd;
    const float * bt = b + (int64_t) t*n_embd;

    ggml_cuda_pdl_sync();

    float sum = 0.0f;
    for (int e = tid; e < n_embd; e += HEPH_CTRL_BLOCK) {
        sum += wr[e]*at[e] + wr[n_embd + e]*bt[e];
    }
    sum = warp_reduce_sum(sum);

    __shared__ float s_sum[HEPH_CTRL_BLOCK / WARP_SIZE];
    if (tid % WARP_SIZE == 0) {
        s_sum[tid / WARP_SIZE] = sum;
    }
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (int k = 0; k < HEPH_CTRL_BLOCK / WARP_SIZE; ++k) {
            total += s_sum[k];
        }
        gate[(int64_t) t*rank + g] = total;
    }
}

// one thread per output element
static __global__ void loop_control_apply_f32(const float * gate, const float * w_forget, const float * w_gain,
                                              const float * cur, const float * msg, float * dst,
                                              const int n_embd, const int rank, const int64_t n) {
    ggml_cuda_pdl_lc();
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    const int     e = idx % n_embd;
    const int64_t t = idx / n_embd;

    const float * gt = gate + t*rank;
    const float * wf = w_forget + (int64_t) e*rank;
    const float * wg = w_gain   + (int64_t) e*rank;

    ggml_cuda_pdl_sync();

    float f = 0.0f;
    float g = 0.0f;
    for (int k = 0; k < rank; ++k) {
        f += wf[k]*gt[k];
        g += wg[k]*gt[k];
    }
    const float c = cur[idx];
    dst[idx] = (c - tanhf(f)*c) + tanhf(g)*msg[idx];
}

void ggml_cuda_op_loop_control(ggml_backend_cuda_context & ctx, const ggml_tensor * a, const ggml_tensor * b,
                               const ggml_tensor * w_down, const ggml_tensor * w_forget, const ggml_tensor * w_gain,
                               const ggml_tensor * cur, const ggml_tensor * msg, ggml_tensor * gate_tmp, ggml_tensor * dst) {
    const int     n_embd = dst->ne[0];
    const int     nt     = dst->ne[1];
    const int     rank   = w_down->ne[1];
    const int64_t n      = (int64_t) n_embd*nt;
    cudaStream_t  stream = ctx.stream();

    {
        const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3(rank, nt, 1), dim3(HEPH_CTRL_BLOCK, 1, 1), 0, stream);
        ggml_cuda_kernel_launch(loop_control_gate_f32, lp, (const float *) a->data, (const float *) b->data,
            (const float *) w_down->data, (float *) gate_tmp->data, n_embd, rank);
    }
    {
        const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3((n + 255) / 256, 1, 1), dim3(256, 1, 1), 0, stream);
        ggml_cuda_kernel_launch(loop_control_apply_f32, lp, (const float *) gate_tmp->data, (const float *) w_forget->data,
            (const float *) w_gain->data, (const float *) cur->data, (const float *) msg->data, (float *) dst->data, n_embd, rank, n);
    }
}

static __global__ void situ_f32(const char * g, const char * u, float * dst, const int n0, const int64_t n,
                                const int64_t g_nb0, const int64_t g_nb1, const int64_t u_nb0, const int64_t u_nb1,
                                const float inv_beta, const float beta, const float inv_lb, const float lb) {
    ggml_cuda_pdl_lc();
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    const int64_t i0 = idx % n0;
    const int64_t i1 = idx / n0;

    ggml_cuda_pdl_sync();

    const float gv = *(const float *) (g + i0*g_nb0 + i1*g_nb1);
    const float uv = *(const float *) (u + i0*u_nb0 + i1*u_nb1);

    // same association as the unfused chain: ((tanh(g/beta)*beta) * sigmoid(g)) * (tanh(u/lb)*lb)
    const float act = (tanhf(gv*inv_beta)*beta) * (1.0f / (1.0f + expf(-gv)));
    dst[idx] = act * (tanhf(uv*inv_lb)*lb);
}

void ggml_cuda_op_situ(ggml_backend_cuda_context & ctx, const ggml_tensor * g, const ggml_tensor * u, ggml_tensor * dst,
                       float inv_beta, float beta, float inv_lb, float lb) {
    const int64_t n  = ggml_nelements(dst);
    const int     n0 = dst->ne[0];
    // rows beyond ne[1] fold into the nb[1] stride only when the higher dims are contiguous in nb[1] steps
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3((n + 255) / 256, 1, 1), dim3(256, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(situ_f32, lp, (const char *) g->data, (const char *) u->data, (float *) dst->data, n0, n,
        (int64_t) g->nb[0], (int64_t) g->nb[1], (int64_t) u->nb[0], (int64_t) u->nb[1], inv_beta, beta, inv_lb, lb);
}

static __global__ void conv_res_scale_f32(const float * x, const float * w, const float * cur, float * dst,
                                          const int n_ch, const int n_conv, const int x_len, const int64_t n, const float scale) {
    ggml_cuda_pdl_lc();
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    const int     c = idx % n_ch;
    const int64_t i = idx / n_ch;

    ggml_cuda_pdl_sync();

    const float * xc = x + (int64_t) c*x_len + i;
    const float * wc = w + (int64_t) c*n_conv;
    float sum = 0.0f;
    for (int j = 0; j < n_conv; ++j) {
        sum += xc[j]*wc[j];
    }
    dst[idx] = (cur[idx] + sum)*scale;
}

void ggml_cuda_op_conv_res_scale(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * w,
                                 const ggml_tensor * cur, ggml_tensor * dst, float scale) {
    const int     n_ch   = w->ne[1];
    const int     n_conv = w->ne[0];
    const int     x_len  = x->ne[0];
    const int64_t n      = ggml_nelements(dst);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3((n + 255) / 256, 1, 1), dim3(256, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(conv_res_scale_f32, lp, (const float *) x->data, (const float *) w->data, (const float *) cur->data,
        (float *) dst->data, n_ch, n_conv, x_len, n, scale);
}
