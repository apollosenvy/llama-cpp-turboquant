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
