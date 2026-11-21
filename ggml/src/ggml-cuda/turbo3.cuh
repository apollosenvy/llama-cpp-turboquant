#pragma once

#include "common.cuh"

#define GGML_TURBO3_TABLE static const __device__
#define GGML_TURBO3_FN static __device__ __forceinline__
#include "ggml-turbo3.h"
#undef GGML_TURBO3_FN
#undef GGML_TURBO3_TABLE

static __device__ void quantize_f32_turbo3_0_block(const float * src, block_turbo3_0 * dst) {
    dst->norm = __float2half(ggml_turbo3_pack(src, dst));
}

static __device__ __forceinline__ void dequantize_turbo3_0(const void * vx, int64_t ib, int iqs, float2 & v) {
    const auto * block = static_cast<const block_turbo3_0 *>(vx) + ib;
    const float norm = __half2float(block->norm);
    v = make_float2(norm*ggml_turbo3_value(block, iqs), norm*ggml_turbo3_value(block, iqs + 1));
}
