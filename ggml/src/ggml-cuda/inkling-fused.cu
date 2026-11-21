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
