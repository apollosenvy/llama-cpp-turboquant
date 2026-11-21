#pragma once

// Three-bit Lloyd-Max centroids for a Gaussian with variance 1/128.
GGML_TURBO3_TABLE float turbo3_centroids[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f,
};

GGML_TURBO3_FN float ggml_turbo3_value(const block_turbo3_0 * block, int index) {
    const int low = (block->qs[index/4] >> (2*(index % 4))) & 3;
    const int high = (block->signs[index/8] >> (index % 8)) & 1;
    return turbo3_centroids[low | (high << 2)];
}

// The attention graph has already rotated each 128-element vector.
GGML_TURBO3_FN float ggml_turbo3_pack(const float * src, block_turbo3_0 * dst) {
    float norm_sq = 0.0f;
    for (int i = 0; i < QK_TURBO3; ++i) {
        norm_sq += src[i]*src[i];
    }
    const float norm = sqrtf(norm_sq);
    const float inv_norm = norm > 0.0f ? 1.0f/norm : 0.0f;
    float recon_sq = 0.0f;
    for (int group = 0; group < QK_TURBO3/8; ++group) {
        unsigned low = 0;
        unsigned high = 0;
        for (int i = 0; i < 8; ++i) {
            const float x = src[8*group + i]*inv_norm;
            const float magnitude = fabsf(x);
            const int level = (magnitude >= 0.043589f) + (magnitude >= 0.091775f) + (magnitude >= 0.154259f);
            const int index = x < 0.0f ? 3 - level : 4 + level;
            low |= (index & 3) << (2*i);
            high |= (index >> 2) << i;
            recon_sq += turbo3_centroids[index]*turbo3_centroids[index];
        }
        dst->qs[2*group]     = (uint8_t) low;
        dst->qs[2*group + 1] = (uint8_t) (low >> 8);
        dst->signs[group]    = (uint8_t) high;
    }
    return norm/sqrtf(recon_sq);
}
