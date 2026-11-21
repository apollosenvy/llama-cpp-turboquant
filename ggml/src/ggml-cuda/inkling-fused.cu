#include "inkling-fused.cuh"
#include "turbo3.cuh"

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

#define HEPH_MIX_BLOCK 1024

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

    // thread s finishes the reduction of state s, thread 0 then does the softmax
    __shared__ float s_w[HEPH_MIX_MAX_STATES]; // rms scale of each state, then its final weight
    __shared__ float s_l[HEPH_MIX_MAX_STATES]; // softmax logits
    if (tid < n_states) {
        float ss = 0.0f;
        float dt = 0.0f;
        for (int k = 0; k < n_warps; ++k) {
            ss += s_ss[tid][k];
            dt += s_dt[tid][k];
        }
        const float r = 1.0f / sqrtf(ss / n_embd + eps);
        s_w[tid] = r;
        s_l[tid] = dt*r*sm_scale;
    }
    __syncthreads();
    if (tid == 0) {
        float max_val = -INFINITY;
        for (int s = 0; s < n_states; ++s) {
            max_val = fmaxf(max_val, s_l[s]);
        }
        float l[HEPH_MIX_MAX_STATES];
        float sum = 0.0f;
        for (int s = 0; s < n_states; ++s) {
            l[s] = expf(s_l[s] - max_val);
            sum += l[s];
        }
        for (int s = 0; s < n_states; ++s) {
            s_w[s] *= l[s] / sum;
        }
    }
    __syncthreads();
    float w[HEPH_MIX_MAX_STATES];
    for (int s = 0; s < n_states; ++s) {
        w[s] = s_w[s];
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

// Loads are latency-bound on this path (about 0.1 us per dependent loop iteration), so the fast variant issues every
// load of a thread at once, keeps the values in registers for the final weighted sum, and has no loop over memory.
#define HEPH_MIX_REG_J 3 // elements per thread: n_embd <= HEPH_MIX_REG_J*HEPH_MIX_BLOCK

template <int S>
static __global__ void mix_states_reg_f32(const heph_mix_states states, const float * query, float * dst,
                                          const int n_embd, const int nq, const float eps, const float sm_scale) {
    ggml_cuda_pdl_lc();
    const int q   = blockIdx.x;
    const int t   = blockIdx.y;
    const int tid = threadIdx.x;

    constexpr int n_warps = HEPH_MIX_BLOCK / WARP_SIZE;
    __shared__ float s_ss[S][n_warps];
    __shared__ float s_dt[S][n_warps];
    __shared__ float s_w[S];
    __shared__ float s_l[S];

    const float * qp = query + (int64_t) q*n_embd;

    ggml_cuda_pdl_sync();

    float qv[HEPH_MIX_REG_J];
    float v[S][HEPH_MIX_REG_J];
#pragma unroll
    for (int j = 0; j < HEPH_MIX_REG_J; ++j) {
        const int e = tid + j*HEPH_MIX_BLOCK;
        qv[j] = e < n_embd ? qp[e] : 0.0f;
    }
#pragma unroll
    for (int s = 0; s < S; ++s) {
        const float * x = states.p[s] + (int64_t) t*n_embd;
#pragma unroll
        for (int j = 0; j < HEPH_MIX_REG_J; ++j) {
            const int e = tid + j*HEPH_MIX_BLOCK;
            v[s][j] = e < n_embd ? x[e] : 0.0f;
        }
    }
#pragma unroll
    for (int s = 0; s < S; ++s) {
        float ss = 0.0f;
        float dt = 0.0f;
#pragma unroll
        for (int j = 0; j < HEPH_MIX_REG_J; ++j) {
            ss += v[s][j]*v[s][j];
            dt += v[s][j]*qv[j];
        }
        ss = warp_reduce_sum(ss);
        dt = warp_reduce_sum(dt);
        if (tid % WARP_SIZE == 0) {
            s_ss[s][tid / WARP_SIZE] = ss;
            s_dt[s][tid / WARP_SIZE] = dt;
        }
    }
    __syncthreads();
    if (tid < S) {
        float ss = 0.0f;
        float dt = 0.0f;
        for (int k = 0; k < n_warps; ++k) {
            ss += s_ss[tid][k];
            dt += s_dt[tid][k];
        }
        const float r = 1.0f / sqrtf(ss / n_embd + eps);
        s_w[tid] = r;
        s_l[tid] = dt*r*sm_scale;
    }
    __syncthreads();
    if (tid == 0) {
        float max_val = -INFINITY;
#pragma unroll
        for (int s = 0; s < S; ++s) {
            max_val = fmaxf(max_val, s_l[s]);
        }
        float l[S];
        float sum = 0.0f;
#pragma unroll
        for (int s = 0; s < S; ++s) {
            l[s] = expf(s_l[s] - max_val);
            sum += l[s];
        }
#pragma unroll
        for (int s = 0; s < S; ++s) {
            s_w[s] *= l[s] / sum;
        }
    }
    __syncthreads();

    float acc[HEPH_MIX_REG_J];
#pragma unroll
    for (int j = 0; j < HEPH_MIX_REG_J; ++j) {
        acc[j] = 0.0f;
    }
#pragma unroll
    for (int s = 0; s < S; ++s) {
        const float w = s_w[s];
#pragma unroll
        for (int j = 0; j < HEPH_MIX_REG_J; ++j) {
            acc[j] += w*v[s][j];
        }
    }
    float * out = dst + ((int64_t) t*nq + q)*n_embd;
#pragma unroll
    for (int j = 0; j < HEPH_MIX_REG_J; ++j) {
        const int e = tid + j*HEPH_MIX_BLOCK;
        if (e < n_embd) {
            out[e] = acc[j];
        }
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
    if (n_embd <= HEPH_MIX_REG_J*HEPH_MIX_BLOCK && n_states <= 8) {
        const float * qd = (const float *) query->data;
        float *       dd = (float *) dst->data;
        switch (n_states) {
            case 1: ggml_cuda_kernel_launch(mix_states_reg_f32<1>, lp, states, qd, dd, n_embd, nq, eps, sm_scale); break;
            case 2: ggml_cuda_kernel_launch(mix_states_reg_f32<2>, lp, states, qd, dd, n_embd, nq, eps, sm_scale); break;
            case 3: ggml_cuda_kernel_launch(mix_states_reg_f32<3>, lp, states, qd, dd, n_embd, nq, eps, sm_scale); break;
            case 4: ggml_cuda_kernel_launch(mix_states_reg_f32<4>, lp, states, qd, dd, n_embd, nq, eps, sm_scale); break;
            case 5: ggml_cuda_kernel_launch(mix_states_reg_f32<5>, lp, states, qd, dd, n_embd, nq, eps, sm_scale); break;
            case 6: ggml_cuda_kernel_launch(mix_states_reg_f32<6>, lp, states, qd, dd, n_embd, nq, eps, sm_scale); break;
            case 7: ggml_cuda_kernel_launch(mix_states_reg_f32<7>, lp, states, qd, dd, n_embd, nq, eps, sm_scale); break;
            case 8: ggml_cuda_kernel_launch(mix_states_reg_f32<8>, lp, states, qd, dd, n_embd, nq, eps, sm_scale); break;
        }
        return;
    }
    ggml_cuda_kernel_launch(mix_states_f32, lp, states, n_states, (const float *) query->data, (float *) dst->data, n_embd, nq, eps, sm_scale);
}

#define HEPH_CTRL_BLOCK 1024
#define HEPH_CTRL_J 3 // elements per thread, loaded at once: n_embd <= HEPH_CTRL_J*HEPH_CTRL_BLOCK

// one block per (gate row, token): gate[g, t] = sum_e w_down[e, g]*a[e, t]*ra + w_down[n + e, g]*b[e, t]*rb
// with_norm: ra and rb are the rms_norm factors of a and b, recomputed per block; otherwise 1
// Launches whose total thread count is a multiple of 98,304 (96 compute units x 1024), or up to 1,536 below one, take
// 6 to 9 us longer on the RX 7900 XTX, up to four multiples (HILLCLIMB.md, "Slow launch grid sizes").
static bool heph_slow_grid(int64_t threads) {
    const int64_t r = threads % 98304;
    return threads <= 4*98304 && (r == 0 || r > 98304 - 1536);
}

template <bool with_norm>
static __global__ void loop_control_gate_f32(const heph_ctrl_jobs jobs, const int n_embd, const int rank) {
    ggml_cuda_pdl_lc();
    const int g   = blockIdx.x;
    const int t   = blockIdx.y;
    const int job = blockIdx.z;
    const int tid = threadIdx.x;
    if (g >= rank) {
        return; // a block the launch added to step over a slow grid size
    }

    const float * wr = jobs.w_down[job] + (int64_t) g*2*n_embd;
    const float * at = jobs.cur[job] + t*jobs.cur_stride[job];
    const float * bt = jobs.msg[job] + t*jobs.msg_stride[job];
    float *       gate  = jobs.gate[job];
    const float   eps_a = jobs.eps_cur[job];
    const float   eps_b = jobs.eps_msg[job];

    ggml_cuda_pdl_sync();

    float av[HEPH_CTRL_J], bv[HEPH_CTRL_J], wa[HEPH_CTRL_J], wb[HEPH_CTRL_J];
#pragma unroll
    for (int j = 0; j < HEPH_CTRL_J; ++j) {
        const int  e  = tid + j*HEPH_CTRL_BLOCK;
        const bool in = e < n_embd;
        av[j] = in ? at[e] : 0.0f;
        bv[j] = in ? bt[e] : 0.0f;
        wa[j] = in ? wr[e] : 0.0f;
        wb[j] = in ? wr[n_embd + e] : 0.0f;
    }
    float v[4] = { 0.0f, 0.0f, 0.0f, 0.0f }; // dot_a, dot_b, ss_a, ss_b
#pragma unroll
    for (int j = 0; j < HEPH_CTRL_J; ++j) {
        v[0] += wa[j]*av[j];
        v[1] += wb[j]*bv[j];
        v[2] += av[j]*av[j];
        v[3] += bv[j]*bv[j];
    }

    constexpr int n_red = with_norm ? 4 : 2;
    __shared__ float s_v[n_red][HEPH_CTRL_BLOCK / WARP_SIZE];
#pragma unroll
    for (int k = 0; k < n_red; ++k) {
        v[k] = warp_reduce_sum(v[k]);
        if (tid % WARP_SIZE == 0) {
            s_v[k][tid / WARP_SIZE] = v[k];
        }
    }
    __syncthreads();
    // second level: the first warp reduces the per-warp partials (HEPH_CTRL_BLOCK / WARP_SIZE == WARP_SIZE)
    if (tid < WARP_SIZE) {
        float tot[n_red];
#pragma unroll
        for (int k = 0; k < n_red; ++k) {
            tot[k] = warp_reduce_sum(s_v[k][tid]);
        }
        if (tid == 0) {
            if constexpr (with_norm) {
                const float ra = 1.0f / sqrtf(tot[2] / n_embd + eps_a);
                const float rb = 1.0f / sqrtf(tot[3] / n_embd + eps_b);
                gate[(int64_t) t*rank + g] = tot[0]*ra + tot[1]*rb;
            } else {
                gate[(int64_t) t*rank + g] = tot[0] + tot[1];
            }
        }
    }
}
static_assert(HEPH_CTRL_BLOCK / WARP_SIZE == WARP_SIZE, "the second reduction level assumes one warp of partials");

static __global__ void loop_control_apply_strided_f32(const heph_ctrl_jobs jobs, const int n_embd, const int rank, const int64_t n);

// all gates in one launch, then all applies in one launch
template <bool with_norm>
static void heph_launch_control(cudaStream_t stream, const heph_ctrl_jobs & jobs, int n_embd, int n_tokens, int rank) {
    {
        const int grid_x = heph_slow_grid((int64_t) rank*n_tokens*jobs.n*HEPH_CTRL_BLOCK) ? rank + 1 : rank;
        const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3(grid_x, n_tokens, jobs.n), dim3(HEPH_CTRL_BLOCK, 1, 1), 0, stream);
        ggml_cuda_kernel_launch(loop_control_gate_f32<with_norm>, lp, jobs, n_embd, rank);
    }
    {
        constexpr int n_waves = 256 / WARP_SIZE; // HEPH_CTRL_APPLY_BLOCK
        const int64_t n      = (int64_t) n_embd*n_tokens;
        // one element per wave for decode, bounded grid for long batches
        const int64_t blocks = std::min<int64_t>((n*jobs.n + n_waves - 1) / n_waves, 16384);
        const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3(blocks, 1, 1), dim3(256, 1, 1), 0, stream);
        ggml_cuda_kernel_launch(loop_control_apply_strided_f32, lp, jobs, n_embd, rank, n);
    }
}

void ggml_cuda_op_loop_control(ggml_backend_cuda_context & ctx, const ggml_tensor * a, const ggml_tensor * b,
                               const ggml_tensor * w_down, const ggml_tensor * w_forget, const ggml_tensor * w_gain,
                               const ggml_tensor * cur, const ggml_tensor * msg, ggml_tensor * gate_tmp, ggml_tensor * dst) {
    const int     n_embd = dst->ne[0];
    const int     nt     = dst->ne[1];
    const int     rank   = w_down->ne[1];
    const int64_t n      = (int64_t) n_embd*nt;
    cudaStream_t  stream = ctx.stream();

    // a and b are the normalised inputs of the gate; the apply reads cur and msg
    heph_ctrl_jobs gate_jobs = {};
    gate_jobs.n = 1;
    gate_jobs.cur[0] = (const float *) a->data;   gate_jobs.cur_stride[0] = n_embd;
    gate_jobs.msg[0] = (const float *) b->data;   gate_jobs.msg_stride[0] = n_embd;
    gate_jobs.w_down[0] = (const float *) w_down->data;
    gate_jobs.gate[0]   = (float *) gate_tmp->data;
    {
        const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3(rank, nt, 1), dim3(HEPH_CTRL_BLOCK, 1, 1), 0, stream);
        ggml_cuda_kernel_launch(loop_control_gate_f32<false>, lp, gate_jobs, n_embd, rank);
    }
    heph_ctrl_jobs jobs = gate_jobs;
    jobs.cur[0] = (const float *) cur->data;
    jobs.msg[0] = (const float *) msg->data;
    jobs.w_forget[0] = (const float *) w_forget->data;
    jobs.w_gain[0]   = (const float *) w_gain->data;
    jobs.dst[0]      = (float *) dst->data;
    {
        constexpr int n_waves = 256 / WARP_SIZE;
        const int64_t blocks = std::min<int64_t>((n + n_waves - 1) / n_waves, 16384);
        const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3(blocks, 1, 1), dim3(256, 1, 1), 0, stream);
        ggml_cuda_kernel_launch(loop_control_apply_strided_f32, lp, jobs, n_embd, rank, n);
    }
}

static __global__ void situ_f32(const char * g, const char * u, float * dst, const int n0, const int n0_pad, const int64_t n,
                                const int64_t g_nb0, const int64_t g_nb1, const int64_t u_nb0, const int64_t u_nb1,
                                const float inv_beta, const float beta, const float inv_lb, const float lb, block_q8_1 * q) {
    ggml_cuda_pdl_lc();
    // rows of n0_pad positions, the last n0_pad - n0 of each are padding (zero q8_1 blocks, nothing else)
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    const int64_t i0 = idx % n0_pad;
    const int64_t i1 = idx / n0_pad;

    ggml_cuda_pdl_sync();

    float v = 0.0f;
    if (i0 < n0) {
        const float gv = *(const float *) (g + i0*g_nb0 + i1*g_nb1);
        const float uv = *(const float *) (u + i0*u_nb0 + i1*u_nb1);

        // same association as the unfused chain: ((tanh(g/beta)*beta) * sigmoid(g)) * (tanh(u/lb)*lb)
        const float act = (tanhf(gv*inv_beta)*beta) * (1.0f / (1.0f + expf(-gv)));
        v = act * (tanhf(uv*inv_lb)*lb);
        dst[i1*n0 + i0] = v;
    }
    if (q) { // as quantize_q8_1: a warp is one block of 32 (n0_pad is a multiple of 32, the thread block of 256)
        const float  amax = warp_reduce_max<QK8_1>(fabsf(v));
        const float  d    = amax / 127.0f;
        const int8_t qv   = amax == 0.0f ? 0 : roundf(v / d);
        const int    sum  = warp_reduce_sum<QK8_1>((int) qv);
        block_q8_1 * blk  = q + idx / QK8_1;
        blk->qs[idx % QK8_1] = qv;
        if (idx % QK8_1 == 0) {
            blk->ds = make_half2(d, d * sum);
        }
    }
}

void ggml_cuda_op_situ(ggml_backend_cuda_context & ctx, const ggml_tensor * g, const ggml_tensor * u, ggml_tensor * dst,
                       float inv_beta, float beta, float inv_lb, float lb, void * q8_1) {
    const int     n0     = dst->ne[0];
    const int     n0_pad = q8_1 ? GGML_PAD(n0, MATRIX_ROW_PADDING) : n0;
    const int64_t n      = ggml_nelements(dst) / n0 * n0_pad;
    // rows beyond ne[1] fold into the nb[1] stride only when the higher dims are contiguous in nb[1] steps
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3((n + 255) / 256, 1, 1), dim3(256, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(situ_f32, lp, (const char *) g->data, (const char *) u->data, (float *) dst->data, n0, n0_pad, n,
        (int64_t) g->nb[0], (int64_t) g->nb[1], (int64_t) u->nb[0], (int64_t) u->nb[1], inv_beta, beta, inv_lb, lb, (block_q8_1 *) q8_1);
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

// One wave per output element: the lanes share the rank-wide dot products, so no thread loops over memory.
// cur and msg are [n_embd, n_tokens] with a column stride.
#define HEPH_CTRL_APPLY_BLOCK 256
static __global__ void loop_control_apply_strided_f32(const heph_ctrl_jobs jobs, const int n_embd, const int rank, const int64_t n) {
    ggml_cuda_pdl_lc();
    constexpr int n_waves = HEPH_CTRL_APPLY_BLOCK / WARP_SIZE;
    const int     lane   = threadIdx.x % WARP_SIZE;
    const int64_t first  = (int64_t) blockIdx.x*n_waves + threadIdx.x / WARP_SIZE;
    const int64_t stride = (int64_t) gridDim.x*n_waves;

    ggml_cuda_pdl_sync();

    // n elements per job, the jobs back to back
    for (int64_t gidx = first; gidx < n*jobs.n; gidx += stride) {
        const int     job = gidx / n;
        const int64_t idx = gidx - job*n;
        const int     e = idx % n_embd;
        const int64_t t = idx / n_embd;

        const float * gt = jobs.gate[job] + t*rank;
        const float * wf = jobs.w_forget[job] + (int64_t) e*rank;
        const float * wg = jobs.w_gain[job]   + (int64_t) e*rank;
        const float * cur = jobs.cur[job];
        const float * msg = jobs.msg[job];
        float *       dst = jobs.dst[job];
        const int64_t cur_stride = jobs.cur_stride[job];
        const int64_t msg_stride = jobs.msg_stride[job];

        float f = 0.0f;
        float g = 0.0f;
        for (int k = lane; k < rank; k += WARP_SIZE) {
            f += wf[k]*gt[k];
            g += wg[k]*gt[k];
        }
        f = warp_reduce_sum(f);
        g = warp_reduce_sum(g);
        if (lane == 0) {
            const float c = cur[t*cur_stride + e];
            dst[idx] = (c - tanhf(f)*c) + tanhf(g)*msg[t*msg_stride + e];
        }
    }
}

void ggml_cuda_op_loop_control_jobs(ggml_backend_cuda_context & ctx, const heph_ctrl_jobs & jobs, int n_embd, int n_tokens, int rank) {
    GGML_ASSERT(jobs.n >= 1 && jobs.n <= HEPH_CTRL_MAX_JOBS && n_embd <= HEPH_CTRL_J*HEPH_CTRL_BLOCK);
    heph_launch_control<true>(ctx.stream(), jobs, n_embd, n_tokens, rank);
}

void ggml_cuda_op_loop_control_raw(ggml_backend_cuda_context & ctx, const ggml_tensor * cur, const ggml_tensor * msg,
                                   float eps_cur, float eps_msg,
                                   const ggml_tensor * w_down, const ggml_tensor * w_forget, const ggml_tensor * w_gain,
                                   ggml_tensor * gate_tmp, ggml_tensor * dst) {
    const int     n_embd = dst->ne[0];
    const int     nt     = dst->ne[1];
    const int     rank   = w_down->ne[1];
    const int64_t n      = (int64_t) n_embd*nt;
    const int64_t cs     = cur->nb[1] / sizeof(float);
    const int64_t ms     = msg->nb[1] / sizeof(float);
    cudaStream_t  stream = ctx.stream();

    heph_ctrl_jobs jobs = {};
    jobs.n = 1;
    jobs.cur[0] = (const float *) cur->data;   jobs.cur_stride[0] = cs;   jobs.eps_cur[0] = eps_cur;
    jobs.msg[0] = (const float *) msg->data;   jobs.msg_stride[0] = ms;   jobs.eps_msg[0] = eps_msg;
    jobs.w_down[0]   = (const float *) w_down->data;
    jobs.w_forget[0] = (const float *) w_forget->data;
    jobs.w_gain[0]   = (const float *) w_gain->data;
    jobs.gate[0]     = (float *) gate_tmp->data;
    jobs.dst[0]      = (float *) dst->data;
    GGML_UNUSED(n);
    heph_launch_control<true>(stream, jobs, n_embd, nt, rank);
}

static __global__ void moe_weights_norm_f32(const float * w, float * dst, const int n_used, const int64_t n_tokens,
                                            const float s, const float b, const float out_scale) {
    ggml_cuda_pdl_lc();
    const int64_t t = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= n_tokens) {
        return;
    }
    ggml_cuda_pdl_sync();
    float sum = 0.0f;
    for (int k = 0; k < n_used; ++k) {
        sum += w[t*n_used + k];
    }
    const float den = sum*s + b;
    for (int k = 0; k < n_used; ++k) {
        dst[t*n_used + k] = w[t*n_used + k] / den * out_scale;
    }
}

void ggml_cuda_op_moe_weights_norm(ggml_backend_cuda_context & ctx, const ggml_tensor * w, ggml_tensor * dst,
                                   float s, float b, float out_scale) {
    const int     n_used   = w->ne[0];
    const int64_t n_tokens = ggml_nelements(w) / n_used;
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3((n_tokens + 63) / 64, 1, 1), dim3(64, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(moe_weights_norm_f32, lp, (const float *) w->data, (float *) dst->data, n_used, n_tokens, s, b, out_scale);
}

static __global__ void rel_bias_mask_f16(const float * bias, const float * r, const float * sinks, const int32_t * rel_pos,
                                         const half * mask, half * dst, const int n_rel, const int n_extent,
                                         const int64_t n_kv, const int64_t n_tokens, const int64_t n,
                                         float * fill, const int fill_n, const float fill_v) {
    ggml_cuda_pdl_lc();
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < fill_n) { // an unrelated small constant fill that follows the chain in the graph rides along
        fill[idx] = fill_v;
    }
    if (idx >= n) {
        return;
    }
    const int64_t k  = idx % n_kv;
    const int64_t th = idx / n_kv;
    const int64_t t  = th % n_tokens;
    const int64_t h  = th / n_tokens;

    ggml_cuda_pdl_sync();

    const int32_t rp = rel_pos[t*n_kv + k];
    float v = 0.0f;
    if (rp >= 0 && rp < n_extent) {
        const float * b  = bias + (int64_t) rp*n_rel;
        const float * rv = r + (t*(n/(n_kv*n_tokens)) + h)*n_rel;
        if (n_rel == 16) { // every load issued at once
            float bv[16], rw[16];
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                bv[j] = b[j];
                rw[j] = rv[j];
            }
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                v += bv[j]*rw[j];
            }
        } else {
            for (int j = 0; j < n_rel; ++j) {
                v += b[j]*rv[j];
            }
        }
    }
    v = v - sinks[t*(n/(n_kv*n_tokens)) + h] + __half2float(mask[t*n_kv + k]);
    dst[idx] = __float2half(v);
}

void ggml_cuda_op_rel_bias_mask(ggml_backend_cuda_context & ctx, const ggml_tensor * bias, const void * r,
                                const void * sinks, const ggml_tensor * rel_pos, const ggml_tensor * mask, ggml_tensor * dst,
                                void * scratch, ggml_tensor * fill, float fill_v) {
    const int     n_rel    = bias->ne[0];
    const int     n_extent = bias->ne[1];
    const int64_t n_kv     = dst->ne[0];
    const int64_t n_tokens = dst->ne[1];
    const int64_t n        = ggml_nelements(dst);
    const int     fill_n   = fill ? (int) ggml_nelements(fill) : 0;
    GGML_ASSERT(fill_n <= n);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3((n + 255) / 256, 1, 1), dim3(256, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(rel_bias_mask_f16, lp, (const float *) bias->data, (const float *) r, (const float *) sinks,
        (const int32_t *) rel_pos->data, (const half *) mask->data, (half *) (scratch ? scratch : dst->data), n_rel, n_extent, n_kv, n_tokens, n,
        fill ? (float *) fill->data : nullptr, fill_n, fill_v);
    if (scratch) {
        // dst shares memory with an input the kernel reads; the kernel wrote to scratch instead
        CUDA_CHECK(cudaMemcpyAsync(dst->data, scratch, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, ctx.stream()));
    }
}

// convolve() head: x4[i, c] = i < n_past ? past[i, c] : cur[c, i - n_past]; writes the concat buffer and the next state
static __global__ void conv_concat_store_f32(const float * past, const int64_t past_stride, const float * cur, float * cat, float * state,
                                             const int n_past, const int n_t, const int n_ch, const int64_t n) {
    ggml_cuda_pdl_lc();
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    const int len = n_past + n_t;
    const int i   = idx % len;
    const int c   = idx / len;

    ggml_cuda_pdl_sync();

    const float v = i < n_past ? past[(int64_t) c*past_stride + i] : cur[(int64_t) (i - n_past)*n_ch + c];
    cat[idx] = v;
    if (i >= n_t) {
        state[(int64_t) c*n_past + (i - n_t)] = v;
    }
}

void ggml_cuda_op_conv_concat_store(ggml_backend_cuda_context & ctx, const ggml_tensor * past, const ggml_tensor * cur,
                                    ggml_tensor * cat, void * state, int n_t) {
    const int     n_past = past->ne[0];
    const int     n_ch   = past->ne[1];
    const int64_t n      = (int64_t) (n_past + n_t)*n_ch;
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3((n + 255) / 256, 1, 1), dim3(256, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(conv_concat_store_f32, lp, (const float *) past->data, (int64_t) (past->nb[1] / sizeof(float)),
        (const float *) cur->data, (float *) cat->data, (float *) state, n_past, n_t, n_ch, n);
}

// whole convolve() without dynamic taps, the concat buffer is never materialised:
// dst[c, t] = (cur[c, t] + sum_j x4[t + j, c]*w[j, c])*scale [+ res[c, t]], state[c, j] = x4[n_t + j, c]
template <int n_conv>
static __global__ void conv_full_f32(const float * past, const int64_t past_stride, const float * cur, const float * w, const float * res,
                                     float * dst, float * state, const int n_t, const int n_ch, const int64_t n, const float scale,
                                     const float * ws_x, const float * ws_w, const int ws_n) {
    ggml_cuda_pdl_lc();
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    constexpr int n_past = n_conv - 1;
    const int c = idx % n_ch;
    const int t = idx / n_ch;

    ggml_cuda_pdl_sync();

    float v[n_conv];
    float wv[n_conv];
#pragma unroll
    for (int j = 0; j < n_conv; ++j) {
        const int i = t + j;
        wv[j] = w[(int64_t) c*n_conv + j];
        if (i < n_past) {
            v[j] = past[(int64_t) c*past_stride + i];
        } else if (ws_x == nullptr) {
            v[j] = cur[(int64_t) (i - n_past)*n_ch + c];
        } else {
            // single token: cur is the weighted sum of the expert outputs, sum_k ws_w[k]*ws_x[c, k]
            float xe[8];
            float we[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                xe[k] = k < ws_n ? ws_x[(int64_t) k*n_ch + c] : 0.0f;
                we[k] = k < ws_n ? ws_w[k] : 0.0f;
            }
            float acc = 0.0f;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                acc += xe[k]*we[k];
            }
            v[j] = acc;
        }
    }
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < n_conv; ++j) {
        sum += v[j]*wv[j];
    }
    const float out = (v[n_past] + sum)*scale; // v[n_past] is cur[c, t]
    dst[idx] = res ? res[idx] + out : out;
    if (t == n_t - 1) {
#pragma unroll
        for (int j = 0; j < n_past; ++j) {
            state[(int64_t) c*n_past + j] = v[j + 1];
        }
    }
}

void ggml_cuda_op_conv_full(ggml_backend_cuda_context & ctx, const ggml_tensor * past, const ggml_tensor * cur, const ggml_tensor * w,
                            const ggml_tensor * res, ggml_tensor * dst, void * state, float scale,
                            const ggml_tensor * ws_x, const ggml_tensor * ws_w) {
    GGML_ASSERT(w->ne[0] == 4);
    GGML_ASSERT(ws_x == nullptr || (ws_w != nullptr && ws_x->ne[1] <= 8 && ggml_nelements(dst) == w->ne[1]));
    const int     n_ch = w->ne[1];
    const int64_t n    = ggml_nelements(dst);
    const int     n_t  = n / n_ch;
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3((n + 255) / 256, 1, 1), dim3(256, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(conv_full_f32<4>, lp, (const float *) past->data, (int64_t) (past->nb[1] / sizeof(float)),
        (const float *) cur->data, (const float *) w->data, res ? (const float *) res->data : nullptr, (float *) dst->data, (float *) state,
        n_t, n_ch, n, scale, ws_x ? (const float *) ws_x->data : nullptr, ws_w ? (const float *) ws_w->data : nullptr,
        ws_x ? (int) ws_x->ne[1] : 0);
}

// Row transform + dense rotation, one block per row of n_rot elements:
//   v   = conv ? (cur + sum_j x4[t + j, c]*w[j, c])*conv_scale : x * (scale ? scale : 1)
//   y   = pre == CLIP ? v*clamp(thr/(||v|| + eps), 0, 1) : pre == RMS ? v/sqrt(mean(v^2) + eps) : v
//   dst = rot^T-style product: dst[i] = sum_k rot[k, i]*y[k]
template <int n_rot, int n_conv>
static __global__ void rot_rows_f32(const heph_rot_args a) {
    ggml_cuda_pdl_lc();
    const int64_t row = blockIdx.x;
    const int     tid = threadIdx.x;
    const int64_t idx = row*n_rot + tid;

    __shared__ float s_y[n_rot];
    __shared__ float s_sum[n_rot / WARP_SIZE];

    ggml_cuda_pdl_sync();

    float v;
    if (a.x4) {
        const int     c = idx % a.n_ch;
        const int64_t t = idx / a.n_ch;
        const float * xc = a.x4 + (int64_t) c*a.x_len + t;
        const float * wc = a.w + (int64_t) c*n_conv;
        float xv[n_conv];
        float wv[n_conv];
#pragma unroll
        for (int j = 0; j < n_conv; ++j) {
            xv[j] = xc[j];
            wv[j] = wc[j];
        }
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < n_conv; ++j) {
            sum += xv[j]*wv[j];
        }
        v = (a.x[idx] + sum)*a.conv_scale;
    } else {
        v = a.x[idx];
        if (a.scale) {
            v *= a.scale[(row % a.scale_rows)*n_rot + tid];
        }
    }

    if (a.pre != HEPH_ROT_PRE_NONE) {
        float ss = warp_reduce_sum(v*v);
        if (tid % WARP_SIZE == 0) {
            s_sum[tid / WARP_SIZE] = ss;
        }
        __syncthreads();
        ss = 0.0f;
#pragma unroll
        for (int k = 0; k < n_rot / WARP_SIZE; ++k) {
            ss += s_sum[k];
        }
        if (a.pre == HEPH_ROT_PRE_CLIP) {
            v *= fminf(fmaxf(a.thr / (sqrtf(ss) + a.eps), 0.0f), 1.0f);
        } else {
            v *= 1.0f / sqrtf(ss / n_rot + a.eps);
        }
    }
    s_y[tid] = v;
    __syncthreads();

    // loads are issued 16 at a time, a dependent load per iteration would cost about 0.1 us each
    const float * rr = a.rot + (int64_t) tid*n_rot;
    float acc = 0.0f;
    for (int k0 = 0; k0 < n_rot; k0 += 16) {
        float r[16];
#pragma unroll
        for (int u = 0; u < 16; ++u) {
            r[u] = rr[k0 + u];
        }
#pragma unroll
        for (int u = 0; u < 16; ++u) {
            acc += r[u]*s_y[k0 + u];
        }
    }
    a.dst[idx] = acc;
}

void ggml_cuda_op_rot_rows(ggml_backend_cuda_context & ctx, const heph_rot_args & a, int64_t n_rows) {
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3(n_rows, 1, 1), dim3(128, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(rot_rows_f32<128, 4>, lp, a);
}

// rms_norm(x)*w for one vector of at most HEPH_NORMQ_J*HEPH_NORMQ_BLOCK elements: writes the F32 result and its q8_1
// form (as quantize_q8_1 would, zero blocks up to n_pad), so the matvecs that read it need no quantize launch.
#define HEPH_NORMQ_BLOCK 1024
#define HEPH_NORMQ_J 3
static __global__ void rms_norm_mul_q8_f32(const heph_normq_jobs jobs, const int n, const int n_pad) {
    ggml_cuda_pdl_lc();
    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int job  = blockIdx.x;

    const float * x   = jobs.x[job];
    const float * w   = jobs.w[job];
    float *       dst = jobs.dst[job];
    block_q8_1 *  q   = (block_q8_1 *) jobs.q8_1[job];
    const float   eps = jobs.eps[job];

    __shared__ float s_part[HEPH_NORMQ_BLOCK / WARP_SIZE];
    __shared__ float s_total;

    ggml_cuda_pdl_sync();

    float xv[HEPH_NORMQ_J];
    float wv[HEPH_NORMQ_J];
#pragma unroll
    for (int j = 0; j < HEPH_NORMQ_J; ++j) {
        const int  e  = tid + j*HEPH_NORMQ_BLOCK;
        const bool in = e < n;
        xv[j] = in ? x[e] : 0.0f;
        wv[j] = in ? w[e] : 0.0f;
    }
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < HEPH_NORMQ_J; ++j) {
        ss += xv[j]*xv[j];
    }
    ss = warp_reduce_sum(ss);
    if (lane == 0) {
        s_part[tid / WARP_SIZE] = ss;
    }
    __syncthreads();
    if (tid < WARP_SIZE) {
        const float total = warp_reduce_sum(s_part[tid]);
        if (tid == 0) {
            s_total = total;
        }
    }
    __syncthreads();
    const float scale = 1.0f / sqrtf(s_total / n + eps);

#pragma unroll
    for (int j = 0; j < HEPH_NORMQ_J; ++j) {
        const int   e = tid + j*HEPH_NORMQ_BLOCK;
        const float v = xv[j]*scale*wv[j];
        if (e < n) {
            dst[e] = v;
        }
        if (e < n_pad) { // the same for a whole warp: a warp is one q8_1 block
            const float amax = warp_reduce_max<QK8_1>(fabsf(v));
            const float d    = amax / 127.0f;
            const int8_t qv  = amax == 0.0f ? 0 : roundf(v / d);
            const int   sum  = warp_reduce_sum<QK8_1>((int) qv);
            q[e / QK8_1].qs[lane] = qv;
            if (lane == 0) {
                q[e / QK8_1].ds = make_half2(d, d * sum);
            }
        }
    }
}
static_assert(HEPH_NORMQ_BLOCK / WARP_SIZE == WARP_SIZE && WARP_SIZE == QK8_1, "one warp per q8_1 block, one warp of partial sums");

void ggml_cuda_op_rms_norm_mul_q8_jobs(ggml_backend_cuda_context & ctx, const heph_normq_jobs & jobs, int n) {
    const int n_pad = GGML_PAD(n, MATRIX_ROW_PADDING);
    GGML_ASSERT(jobs.n >= 1 && jobs.n <= HEPH_NORMQ_MAX_JOBS && n <= HEPH_NORMQ_J*HEPH_NORMQ_BLOCK && n_pad <= HEPH_NORMQ_J*HEPH_NORMQ_BLOCK);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3(jobs.n, 1, 1), dim3(HEPH_NORMQ_BLOCK, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(rms_norm_mul_q8_f32, lp, jobs, n, n_pad);
}

void ggml_cuda_op_rms_norm_mul_q8(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * w, ggml_tensor * dst,
                                  void * q8_1, float eps) {
    heph_normq_jobs jobs = {};
    jobs.n       = 1;
    jobs.x[0]    = (const float *) x->data;
    jobs.w[0]    = (const float *) w->data;
    jobs.dst[0]  = (float *) dst->data;
    jobs.q8_1[0] = q8_1;
    jobs.eps[0]  = eps;
    ggml_cuda_op_rms_norm_mul_q8_jobs(ctx, jobs, ggml_nelements(dst));
}

static __global__ void row_jobs_f32(const heph_row_jobs jobs) {
    constexpr int n_rot  = 128;
    constexpr int n_conv = 4;
    constexpr int n_past = n_conv - 1;

    ggml_cuda_pdl_lc();
    int      ji     = 0;
    uint32_t block0 = 0;
#pragma unroll
    for (int k = 0; k < HEPH_ROW_MAX_JOBS; ++k) {
        if (k < jobs.n && blockIdx.x >= jobs.job[k].block_end) {
            ji     = k + 1;
            block0 = jobs.job[k].block_end;
        }
    }
    if (ji >= jobs.n) {
        return;
    }
    const heph_row_job & j = jobs.job[ji];

    const int     tid    = threadIdx.x;
    const int64_t row    = blockIdx.x - block0;
    const int64_t idx    = row*j.width + tid;
    const bool    active = tid < j.width;

    __shared__ float s_y[n_rot];
    __shared__ float s_sum[n_rot / WARP_SIZE];

    ggml_cuda_pdl_sync();

    float v = 0.0f;
    if (active) {
        if (j.past) {
            const int     c = idx % j.n_ch;
            const int64_t t = idx / j.n_ch;
            float xv[n_conv];
            float wv[n_conv];
#pragma unroll
            for (int k = 0; k < n_conv; ++k) {
                const int64_t i = t + k;
                xv[k] = i < n_past ? j.past[(int64_t) c*j.past_stride + i] : j.x[(i - n_past)*j.n_ch + c];
                wv[k] = j.w[(int64_t) c*n_conv + k];
            }
            if (j.dyn) {
                // lag l of the taps weighs the input n_conv - 1 - l steps back, the same tap for the 128 channels of a head
                const float * dt = j.dyn + t*j.dyn_stride + (int64_t) (c / n_rot)*n_conv;
#pragma unroll
                for (int k = 0; k < n_conv; ++k) {
                    wv[k] += dt[n_conv - 1 - k];
                }
            }
            float sum = 0.0f;
#pragma unroll
            for (int k = 0; k < n_conv; ++k) {
                sum += xv[k]*wv[k];
            }
            v = (xv[n_past] + sum)*j.conv_scale; // xv[n_past] is cur[c, t]
            if (t == j.n_t - 1) {
#pragma unroll
                for (int k = 0; k < n_past; ++k) {
                    j.state[(int64_t) c*n_past + k] = xv[k + 1];
                }
            }
        } else {
            v = j.x[idx];
            if (j.scale) {
                v *= j.scale[(row % j.scale_rows)*j.width + tid];
            }
        }
    }

    if (j.pre != HEPH_ROT_PRE_NONE) {
        float ss = warp_reduce_sum(v*v);
        if (tid % WARP_SIZE == 0) {
            s_sum[tid / WARP_SIZE] = ss;
        }
        __syncthreads();
        ss = 0.0f;
#pragma unroll
        for (int k = 0; k < n_rot / WARP_SIZE; ++k) {
            ss += s_sum[k];
        }
        if (j.pre == HEPH_ROT_PRE_CLIP) {
            v *= fminf(fmaxf(j.thr / (sqrtf(ss) + j.eps), 0.0f), 1.0f);
        } else {
            v *= 1.0f / sqrtf(ss / j.width + j.eps);
        }
    }
    if (j.post) {
        v *= j.post[row / j.rows_per_token];
    }
    if (j.rot == nullptr) {
        if (active) {
            j.dst[idx] = v;
        }
        return;
    }
    s_y[tid] = v;
    __syncthreads();

    const float * rr = j.rot + (int64_t) tid*n_rot;
    float acc = 0.0f;
    for (int k0 = 0; k0 < n_rot; k0 += 16) {
        float r[16];
#pragma unroll
        for (int u = 0; u < 16; ++u) {
            r[u] = rr[k0 + u];
        }
#pragma unroll
        for (int u = 0; u < 16; ++u) {
            acc += r[u]*s_y[k0 + u];
        }
    }
    if (j.gate) {
        acc *= 1.0f / (1.0f + expf(-j.gate[idx]));
    }
    if (!(j.t3 && j.skip_f32)) {
        j.dst[idx] = acc;
    }
    if (j.q8) { // as quantize_q8_1: a warp is one block of 32
        const float  amax = warp_reduce_max<QK8_1>(fabsf(acc));
        const float  d    = amax / 127.0f;
        const int8_t qv   = amax == 0.0f ? 0 : roundf(acc / d);
        const int    sum  = warp_reduce_sum<QK8_1>((int) qv);
        block_q8_1 * q    = (block_q8_1 *) j.q8 + idx / QK8_1;
        q->qs[tid % QK8_1] = qv;
        if (tid % QK8_1 == 0) {
            q->ds = make_half2(d, d * sum);
        }
    }
    if (j.t3 == nullptr) {
        return;
    }

    // ggml_turbo3_pack() over the block: the row is one 128-element cache block
    __shared__ uint8_t s_code[n_rot];
    const int lane = tid % WARP_SIZE;
    float part = warp_reduce_sum(acc*acc);
    if (lane == 0) {
        s_sum[tid / WARP_SIZE] = part;
    }
    __syncthreads();
    float norm_sq = 0.0f;
#pragma unroll
    for (int k = 0; k < n_rot / WARP_SIZE; ++k) {
        norm_sq += s_sum[k];
    }
    const float norm      = sqrtf(norm_sq);
    const float inv_norm  = norm > 0.0f ? 1.0f/norm : 0.0f;
    const float xn        = acc*inv_norm;
    const float magnitude = fabsf(xn);
    const int   level     = (magnitude >= 0.043589f) + (magnitude >= 0.091775f) + (magnitude >= 0.154259f);
    const int   code      = xn < 0.0f ? 3 - level : 4 + level;
    s_code[tid] = (uint8_t) code;
    part = warp_reduce_sum(turbo3_centroids[code]*turbo3_centroids[code]);
    __syncthreads(); // every thread has read the norm partials
    if (lane == 0) {
        s_sum[tid / WARP_SIZE] = part;
    }
    __syncthreads();

    const int64_t t    = row / j.t3_heads;
    const int     head = row % j.t3_heads;
    block_turbo3_0 * blk = (block_turbo3_0 *) ((char *) j.t3 + j.t3_idx[t]*j.t3_row_bytes) + head;
    if (tid < QK_TURBO3/4) {
        unsigned low = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            low |= (s_code[4*tid + k] & 3u) << (2*k);
        }
        blk->qs[tid] = (uint8_t) low;
    }
    if (tid < QK_TURBO3/8) {
        unsigned high = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            high |= (unsigned) (s_code[8*tid + k] >> 2) << k;
        }
        blk->signs[tid] = (uint8_t) high;
    }
    if (tid == 0) {
        float recon_sq = 0.0f;
#pragma unroll
        for (int k = 0; k < n_rot / WARP_SIZE; ++k) {
            recon_sq += s_sum[k];
        }
        blk->norm = __float2half(norm/sqrtf(recon_sq));
    }
}
static_assert(QK_TURBO3 == 128, "a row job stores one turbo3 block per row");

void ggml_cuda_op_row_jobs(ggml_backend_cuda_context & ctx, const heph_row_jobs & jobs) {
    GGML_ASSERT(jobs.n >= 1 && jobs.n <= HEPH_ROW_MAX_JOBS);
    const uint32_t blocks = jobs.job[jobs.n - 1].block_end;
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3(blocks, 1, 1), dim3(128, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(row_jobs_f32, lp, jobs);
}

#define HEPH_ROUTE_BLOCK 128
#define HEPH_ROUTE_MAX_USED 16
// one block per token, one thread per expert: every thread ranks its own score against the others from shared memory
static __global__ void moe_route_f32(const float * logits, const float * bias, int32_t * ids, float * w_out,
                                     const int n_expert, const int n_used, const int64_t ids_stride,
                                     const float s, const float b, const float out_scale) {
    ggml_cuda_pdl_lc();
    const int64_t t   = blockIdx.x;
    const int     tid = threadIdx.x;

    __shared__ float s_sel[HEPH_ROUTE_BLOCK];
    __shared__ float s_w[HEPH_ROUTE_MAX_USED];

    ggml_cuda_pdl_sync();

    float p   = 0.0f;
    float sel = -INFINITY;
    if (tid < n_expert) {
        p   = 1.0f / (1.0f + expf(-logits[t*n_expert + tid]));
        sel = bias ? p + bias[tid] : p;
    }
    s_sel[tid] = sel;
    __syncthreads();

    int rank = 0;
    for (int j0 = 0; j0 < HEPH_ROUTE_BLOCK; j0 += 16) {
#pragma unroll
        for (int u = 0; u < 16; ++u) {
            const int   j     = j0 + u;
            const float other = s_sel[j];
            rank += (other > sel) || (other == sel && j < tid);
        }
    }
    if (tid < n_expert && rank < n_used) {
        ids[t*ids_stride + rank] = tid;
        s_w[rank] = p;
    }
    __syncthreads();
    if (tid < n_used) {
        float sum = 0.0f;
        for (int k = 0; k < n_used; ++k) {
            sum += s_w[k];
        }
        w_out[t*n_used + tid] = s_w[tid] / (sum*s + b) * out_scale;
    }
}

void ggml_cuda_op_moe_route(ggml_backend_cuda_context & ctx, const ggml_tensor * logits, const ggml_tensor * bias,
                            void * ids, int64_t ids_stride, ggml_tensor * weights, int n_used, float s, float b, float out_scale) {
    const int     n_expert = logits->ne[0];
    const int64_t n_tokens = ggml_nelements(logits) / n_expert;
    GGML_ASSERT(n_expert <= HEPH_ROUTE_BLOCK && n_used <= HEPH_ROUTE_MAX_USED && n_used <= n_expert);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3(n_tokens, 1, 1), dim3(HEPH_ROUTE_BLOCK, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(moe_route_f32, lp, (const float *) logits->data, bias ? (const float *) bias->data : nullptr,
        (int32_t *) ids, (float *) weights->data, n_expert, n_used, ids_stride, s, b, out_scale);
}
