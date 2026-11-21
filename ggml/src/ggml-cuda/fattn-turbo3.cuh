#pragma once

// Flash attention over a turbo3 K/V cache with head size 128 on RDNA, for one token (1 or 3 query columns per block).
//
// The generic vector kernel spends its time on instruction count, not on memory: per 128-key tile and wave it runs
// about 2,800 instructions with one column, of which the 3-bit decode is a small part. The rest is 64-bit address
// arithmetic for byte loads, shuffles through LDS and waits. This kernel does the same computation with
//   - loads of whole code words off a scalar base (32-bit lane offsets), all of a tile issued before any is used,
//   - one decode of 16 values per (qs word, signs word), shared by all columns,
//   - sums and broadcasts across the 8 lanes of a key done in registers (DPP), nothing through LDS in the loop,
//   - the K and V norms folded into the per-key scalar instead of the per-value products.
// Layout: a block is 4 waves of 32 lanes and takes tiles of 128 keys. Lanes work in groups of 8 on one key at a time:
// lane l of a group owns values 16*l .. 16*l+15 of the key, for K (dot product with Q) and for V (accumulation).
// Lane t of the block keeps the score of key t of the tile.
//
// Tried and dropped: skipping the V rows of a wave whose 32 weights are all zero as halves. It is exact, but this
// model's clipped logits only give zero weights for masked keys, and a 104K-key decode ran no faster with it.

#include "common.cuh"
#include "fattn-common.cuh"

#ifdef GGML_USE_HIP
#define GGML_CUDA_FATTN_TURBO3_AVAILABLE // host side: the kernel exists; whether to use it is decided from the device

// RDNA and the dot product instruction are only known in the device compilation
#if defined(RDNA) && defined(V_DOT2_F32_F16_AVAILABLE)
#define GGML_CUDA_FATTN_TURBO3_DEVICE

template <typename T>
static __device__ __forceinline__ T t3_load(const char * p) {
    T v;
    __builtin_memcpy(&v, p, sizeof(T));
    return v;
}

// value of another lane, permuted in registers. ctrl: 0x141 mirror within 8 lanes, 0xB1 swap neighbours,
// 0x4E swap pairs, 0x128 rotate a row of 16 lanes by 8
template <unsigned ctrl>
static __device__ __forceinline__ unsigned t3_dpp(const unsigned x) {
    return __builtin_amdgcn_mov_dpp(x, ctrl, 0xf, 0xf, false);
}

template <unsigned ctrl>
static __device__ __forceinline__ float t3_dpp(const float x) {
    return __builtin_bit_cast(float, t3_dpp<ctrl>(__builtin_bit_cast(unsigned, x)));
}

// sum over each group of 8 neighbouring lanes
static __device__ __forceinline__ float t3_sum8(float x) {
    x += t3_dpp<0x141>(x);
    x += t3_dpp<0xB1>(x);
    x += t3_dpp<0x4E>(x);
    return x;
}

// maximum over the wave
static __device__ __forceinline__ float t3_max32(float x) {
    x = fmaxf(x, t3_dpp<0x141>(x));
    x = fmaxf(x, t3_dpp<0xB1>(x));
    x = fmaxf(x, t3_dpp<0x4E>(x));
    x = fmaxf(x, t3_dpp<0x128>(x));
    return fmaxf(x, ggml_cuda_shfl_xor(x, 16, WARP_SIZE));
}

// sum over the wave
static __device__ __forceinline__ float t3_sum32(float x) {
    x = t3_sum8(x);
    x += t3_dpp<0x128>(x);
    return x + ggml_cuda_shfl_xor(x, 16, WARP_SIZE);
}

// lane i of each group of 8 lanes, in all 8 lanes of the group
template <int i>
static __device__ __forceinline__ unsigned t3_bcast8_lane(const unsigned x) {
    constexpr unsigned q = i & 3;
    const unsigned y = __builtin_amdgcn_mov_dpp(x, q | q << 2 | q << 4 | q << 6, 0xf, 0xf, false);
    // the other four lanes of the group copy it: row shift by 4 lanes, written to every second quad only
    if constexpr (i < 4) {
        return __builtin_amdgcn_update_dpp(y, y, 0x114, 0xf, 0xA, false);
    } else {
        return __builtin_amdgcn_update_dpp(y, y, 0x104, 0xf, 0x5, false);
    }
}

static __device__ __forceinline__ unsigned t3_bcast8(const unsigned x, const int i) {
    switch (i) {
        case 0:  return t3_bcast8_lane<0>(x);
        case 1:  return t3_bcast8_lane<1>(x);
        case 2:  return t3_bcast8_lane<2>(x);
        case 3:  return t3_bcast8_lane<3>(x);
        case 4:  return t3_bcast8_lane<4>(x);
        case 5:  return t3_bcast8_lane<5>(x);
        case 6:  return t3_bcast8_lane<6>(x);
        default: return t3_bcast8_lane<7>(x);
    }
}

// The 16 centroids of one qs word (16 two-bit codes) and its 16 sign bits.
// c[2*k] holds values k and k+4, c[2*k + 1] values k+8 and k+12 (k = 0..3): values k, k+4, k+8, k+12 sit at the same
// bit offset of the four qs bytes, so one shift and one mask give their four table indices.
static __device__ __forceinline__ void turbo3_centroids_half16(const unsigned qs, const unsigned signs, half2 * c) {
    const unsigned t  = (signs | (signs << 8)) & 0x00ff00ffu;
    const unsigned s4 = (t << 2) | (t << 6); // bit 2+k of byte b: sign bit of value 4*b + k (other bits are not used)
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const unsigned indices = ((qs >> (2*k)) & 0x03030303u) | ((s4 >> k) & 0x04040404u);

        // Byte tables for the eight FP16 Lloyd-Max centroids, in quantization index order.
        const unsigned low_parts  = __builtin_amdgcn_perm(0x1a8b357eu, 0x7e358b1au, indices);
        const unsigned high_parts = __builtin_amdgcn_perm(0x322f2c25u, 0xa5acafb2u, indices);
        c[2*k + 0] = __builtin_bit_cast(half2, __builtin_amdgcn_perm(high_parts, low_parts, 0x05010400u));
        c[2*k + 1] = __builtin_bit_cast(half2, __builtin_amdgcn_perm(high_parts, low_parts, 0x07030602u));
    }
}

#endif // defined(RDNA) && defined(V_DOT2_F32_F16_AVAILABLE)

// position of half comp of c[r] (see turbo3_centroids_half16) within the 16 values of a lane
static constexpr __device__ int t3_value_index(const int r, const int comp) {
    return (r >> 1) + 4*(2*(r & 1) + comp);
}

template <int ncols>
__launch_bounds__(128, 1)
static __global__ void flash_attn_ext_turbo3(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        const int  * KV_rows_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int64_t nb32, const int64_t nb33) {
    ggml_cuda_pdl_lc();
#ifdef GGML_CUDA_FATTN_TURBO3_DEVICE
    GGML_UNUSED_VARS(KV_rows_ptr, logit_softcap, ne00, ne03, ne10, ne13, ne31);

    constexpr int D        = QK_TURBO3;
    constexpr int nthreads = 128;
    constexpr int nwarps   = nthreads / WARP_SIZE;
    constexpr int ngroup   = 8;          // lanes per key
    constexpr int nreg     = D/(2*ngroup); // half2 registers per lane and column
    static_assert(WARP_SIZE == 32 && nwarps == 4 && nreg == 8 && ncols <= nwarps, "layout is fixed");

    const int ic0      = blockIdx.x * ncols;
    const int sequence = blockIdx.z / ne02;
    const int head     = blockIdx.z - sequence*ne02;
    const int gqa_ratio = ne02 / ne12;

    const char * Q = Q_ptr + nb03*sequence + nb02* head              + nb01*ic0;
    const char * K = K_ptr + nb13*sequence + nb12*(head / gqa_ratio);
    const char * V = V_ptr + nb23*sequence + nb22*(head / gqa_ratio);
    const char * maskh = mask_ptr ? mask_ptr + nb33*(sequence % ne33) + nb32*(head % ne32) + nb31*ic0 : nullptr;

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    const uint32_t tid = WARP_SIZE*threadIdx.y + threadIdx.x; // key of the tile whose score this lane keeps
    const uint32_t l8  = threadIdx.x % ngroup;                // values 16*l8 .. 16*l8 + 15
    const uint32_t g0  = tid - l8;                            // first key of this lane's group
    __builtin_assume(tid < nthreads);

    // Q in registers, scaled, in the pairing turbo3_centroids_half16 produces
    half2 Q_reg[ncols][nreg];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        const float * Q_j = (const float *) (Q + j*nb01) + 2*nreg*l8;
        float q[2*nreg];
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            ggml_cuda_memcpy_1<4*sizeof(float)>(q + 4*b, Q_j + 4*b);
        }
#pragma unroll
        for (int r = 0; r < nreg; ++r) {
            Q_reg[j][r] = make_half2(q[t3_value_index(r, 0)]*scale, q[t3_value_index(r, 1)]*scale);
        }
    }

    half2 VKQ[ncols][nreg];
    float KQ_max[ncols];
    float KQ_sum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
#pragma unroll
        for (int r = 0; r < nreg; ++r) {
            VKQ[j][r] = make_half2(0.0f, 0.0f);
        }
    }

    ggml_cuda_pdl_sync();

    // 32-bit lane offsets from the first row of a tile; the row itself is a scalar, so loads are base + lane offset
    const uint32_t off_K_qs    = g0*uint32_t(nb11) + offsetof(block_turbo3_0, qs)    + 4*l8;
    const uint32_t off_K_signs = g0*uint32_t(nb11) + offsetof(block_turbo3_0, signs) + 2*l8;
    const uint32_t off_K_norm  = tid*uint32_t(nb11) + offsetof(block_turbo3_0, norm);
    const uint32_t off_V_qs    = g0*uint32_t(nb21) + offsetof(block_turbo3_0, qs)    + 4*l8;
    const uint32_t off_V_signs = g0*uint32_t(nb21) + offsetof(block_turbo3_0, signs) + 2*l8;
    const uint32_t off_V_norm  = tid*uint32_t(nb21) + offsetof(block_turbo3_0, norm);
    const uint32_t off_mask    = tid*uint32_t(sizeof(half));

    const int k_VKQ_max = KV_max_ptr ? KV_max_ptr[sequence*gridDim.x + blockIdx.x] : ne11;
    const size_t tile_step_K = size_t(gridDim.y)*nthreads*nb11;
    const size_t tile_step_V = size_t(gridDim.y)*nthreads*nb21;
    const size_t tile_step_m = size_t(gridDim.y)*nthreads*sizeof(half);
    K += size_t(blockIdx.y)*nthreads*nb11;
    V += size_t(blockIdx.y)*nthreads*nb21;
    if (maskh) {
        maskh += size_t(blockIdx.y)*nthreads*sizeof(half);
    }

    for (int k_VKQ_0 = blockIdx.y*nthreads; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nthreads,
             K += tile_step_K, V += tile_step_V, maskh += tile_step_m) {

        // the code words of the group's 8 keys
        unsigned k_qs[ngroup];
        unsigned k_signs[ngroup];
#pragma unroll
        for (int i = 0; i < ngroup; ++i) {
            const char * row = K + size_t(i)*nb11;
            k_qs[i]    = t3_load<uint32_t>(row + off_K_qs);
            k_signs[i] = t3_load<uint16_t>(row + off_K_signs);
        }
        const float norm_K = __half2float(t3_load<half>(K + off_K_norm));
        const float norm_V = __half2float(t3_load<half>(V + off_V_norm));

        float KQ_reg[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            KQ_reg[j] = 0.0f;
        }

#pragma unroll
        for (int i = 0; i < ngroup; ++i) {
            half2 c[nreg];
            turbo3_centroids_half16(k_qs[i], k_signs[i], c);
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum = 0.0f;
#pragma unroll
                for (int r = 0; r < nreg; ++r) {
                    ggml_cuda_mad(sum, c[r], Q_reg[j][r]);
                }
                sum = t3_sum8(sum);
                KQ_reg[j] = l8 == uint32_t(i) ? sum : KQ_reg[j];
            }
        }

        unsigned weight[ncols]; // softmax weight times the V norm of this lane's key, as two halves
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            KQ_reg[j] *= norm_K;
            if (maskh) {
                KQ_reg[j] += slope*__half2float(t3_load<half>(maskh + size_t(j)*ne11*sizeof(half) + off_mask));
            }

            const float KQ_max_new = fmaxf(KQ_max[j], t3_max32(KQ_reg[j] + FATTN_KQ_MAX_OFFSET));
            if (KQ_max_new > KQ_max[j]) { // the same for every lane of the wave
                const float KQ_max_scale = expf(KQ_max[j] - KQ_max_new);
                KQ_max[j]  = KQ_max_new;
                KQ_sum[j] *= KQ_max_scale;
                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
                for (int r = 0; r < nreg; ++r) {
                    VKQ[j][r] *= KQ_max_scale_h2;
                }
            }

            KQ_reg[j]  = expf(KQ_reg[j] - KQ_max[j]);
            KQ_sum[j] += KQ_reg[j];

            const float w = KQ_reg[j]*norm_V;
            weight[j] = __builtin_bit_cast(unsigned, make_half2(w, w));
        }

        unsigned v_qs[ngroup];
        unsigned v_signs[ngroup];
#pragma unroll
        for (int i = 0; i < ngroup; ++i) {
            const char * row = V + size_t(i)*nb21;
            v_qs[i]    = t3_load<uint32_t>(row + off_V_qs);
            v_signs[i] = t3_load<uint16_t>(row + off_V_signs);
        }

#pragma unroll
        for (int i = 0; i < ngroup; ++i) {
            half2 c[nreg];
            turbo3_centroids_half16(v_qs[i], v_signs[i], c);
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                const half2 w = __builtin_bit_cast(half2, t3_bcast8(weight[j], i));
#pragma unroll
                for (int r = 0; r < nreg; ++r) {
                    VKQ[j][r] += c[r]*w;
                }
            }
        }
    }

    if (sinks_ptr && blockIdx.y == 0) {
        const float sink = ((const float *) sinks_ptr)[head];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            if (int(threadIdx.y) != j) { // one wave per column adds the sink
                continue;
            }
            const float kqmax_new_j = fmaxf(sink, KQ_max[j]);
            const float KQ_max_scale = expf(KQ_max[j] - kqmax_new_j);
            KQ_max[j] = kqmax_new_j;

            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + (threadIdx.x == 0 ? expf(sink - KQ_max[j]) : 0.0f);

            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int r = 0; r < nreg; ++r) {
                VKQ[j][r] *= KQ_max_scale_h2;
            }
        }
    }

    // per wave: the partial result (sum of its four lane groups), its maximum and its sum of weights
    __shared__ half  VKQ_shared[ncols][nwarps][D];
    __shared__ float KQ_max_shared[ncols][nwarps];
    __shared__ float KQ_sum_shared[ncols][nwarps];

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        const float sum = t3_sum32(KQ_sum[j]);
        if (threadIdx.x == 0) {
            KQ_max_shared[j][threadIdx.y] = KQ_max[j];
            KQ_sum_shared[j][threadIdx.y] = sum;
        }
#pragma unroll
        for (int r = 0; r < nreg; ++r) {
            half2 v = VKQ[j][r];
            v += __builtin_bit_cast(half2, t3_dpp<0x128>(__builtin_bit_cast(unsigned, v)));
            v += __builtin_bit_cast(half2, ggml_cuda_shfl_xor(__builtin_bit_cast(unsigned, v), 16, WARP_SIZE));
            if (threadIdx.x < ngroup) {
                half * out = &VKQ_shared[j][threadIdx.y][2*nreg*l8];
                out[t3_value_index(r, 0)] = __low2half(v);
                out[t3_value_index(r, 1)] = __high2half(v);
            }
        }
    }

    __syncthreads();

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        float kqmax = KQ_max_shared[j][0];
#pragma unroll
        for (int w = 1; w < nwarps; ++w) {
            kqmax = fmaxf(kqmax, KQ_max_shared[j][w]);
        }

        float dst_val = 0.0f;
        float kqsum   = 0.0f;
#pragma unroll
        for (int w = 0; w < nwarps; ++w) {
            const float kqmax_scale = expf(KQ_max_shared[j][w] - kqmax);
            dst_val += kqmax_scale*__half2float(VKQ_shared[j][w][tid]);
            kqsum   += kqmax_scale*KQ_sum_shared[j][w];
        }

        if (gridDim.y == 1) {
            dst_val /= kqsum;
        }
        const int j_dst = (sequence*int(ne01.z) + ic0 + j)*ne02 + head;
        dst_ptr[(size_t(j_dst)*gridDim.y + blockIdx.y)*D + tid] = dst_val;
        if (gridDim.y != 1 && tid == uint32_t(j)) {
            dst_meta_ptr[size_t(j_dst)*gridDim.y + blockIdx.y] = make_float2(kqmax, kqsum);
        }
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, KV_rows_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // GGML_CUDA_FATTN_TURBO3_DEVICE
}

#endif // GGML_USE_HIP
