// probe-pad-k-chain: numeric conviction instrument for the padded-K turbo3 chain.
//
// The KL gate showed gpt-oss (head 64 -> pad 128) K-side turbo3 destroys the
// output distribution (median KLD 0.735) while muse (native 128) is healthy
// (0.018), on CPU and HIP alike. This probe replays the exact graph chain a
// padded K row travels and checks each stage against analytic truth:
//
//   leg A: dot(wht(pad q), wht(pad k))  vs dot(q,k)  — rotation orthonormality,
//          must match to fp error; failure => the WHT op itself.
//   leg B: dot(wht(pad q), dequant(set_rows_turbo3(pad k))) vs dot(q,k)
//          — the full write+read chain; failure at >> quant noise while A and C
//          pass => rotation DISAGREEMENT between graph wht and quantizer wht.
//   leg C: NMSE(dequant(set_rows_turbo3(pad k)), wht(pad k)) — quantizer
//          fidelity in rotated space; ~0.02 expected for 3-bit turbo3.
//
// Shapes: 64->128 (gpt-oss), 576->640 (deepseek4 MLA), 128 native (control,
// the muse-proven path — the probe itself is validated by this leg passing).
#include "ggml.h"
#include "ggml-cpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float dotf(const float * a, const float * b, int n) {
    double s = 0;
    for (int i = 0; i < n; i++) s += (double) a[i] * b[i];
    return (float) s;
}

static float nmse(const float * a, const float * ref, int n) {
    double e = 0, r = 0;
    for (int i = 0; i < n; i++) { double d = a[i] - ref[i]; e += d*d; r += (double) ref[i]*ref[i]; }
    return (float) (e / (r > 0 ? r : 1));
}

// one probe at head_dim d (padded up to pd, multiples of 128)
static int probe(int d, unsigned seed) {
    const int pd = ((d + 127) / 128) * 128;
    printf("== head_dim %d -> padded %d\n", d, pd);

    struct ggml_init_params ip = { .mem_size = 256*1024*1024, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, 1);
    struct ggml_tensor * k = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, 1);
    srand(seed);
    for (int i = 0; i < d; i++) {
        ((float *) q->data)[i] = (float) rand() / RAND_MAX * 2.0f - 1.0f;
        ((float *) k->data)[i] = (float) rand() / RAND_MAX * 2.0f - 1.0f;
    }
    const float ref_dot = dotf((float *) q->data, (float *) k->data, d);

    // graph-side chain, exactly as build_attn does for Q and cpy_k does for K
    struct ggml_tensor * qp = q, * kp = k;
    if (d != pd) {
        qp = ggml_pad(ctx, q, pd - d, 0, 0, 0);
        kp = ggml_pad(ctx, k, pd - d, 0, 0, 0);
    }
    qp = ggml_cont(ctx, qp);
    kp = ggml_cont(ctx, kp);
    struct ggml_tensor * q_rot = ggml_turbo_wht(ctx, qp, 0, 0, NULL);   // auto group, like build_attn
    struct ggml_tensor * k_rot = ggml_turbo_wht(ctx, kp, 0, 0, NULL);   // reference rotation of K

    // write side: set_rows into a turbo3 cache row, group forced to 128 like cpy_k
    struct ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_TURBO3_0, pd, 4);
    struct ggml_tensor * idx   = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
    ((int64_t *) idx->data)[0] = 0;
    struct ggml_tensor * wr = ggml_set_rows(ctx, cache, kp, idx);
    { int32_t g = 128; memcpy(wr->op_params, &g, sizeof g); }

    // read back: dequant the cache (rotated space, no inverse — mirrors FA consumption)
    struct ggml_tensor * back = ggml_cpy(ctx, cache, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, pd, 4));

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, q_rot);
    ggml_build_forward_expand(gf, k_rot);
    ggml_build_forward_expand(gf, wr);
    ggml_build_forward_expand(gf, back);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    const float * Qr = (const float *) q_rot->data;
    const float * Kr = (const float *) k_rot->data;
    const float * Kq = (const float *) back->data;   // row 0 = our written row

    const float dot_rot   = dotf(Qr, Kr, pd);
    const float dot_quant = dotf(Qr, Kq, pd);
    const float c_nmse    = nmse(Kq, Kr, pd);

    const float a_err = fabsf(dot_rot - ref_dot)   / fmaxf(fabsf(ref_dot), 1e-6f);
    const float b_err = fabsf(dot_quant - ref_dot) / fmaxf(fabsf(ref_dot), 1e-6f);

    printf("  ref dot        %+.6f\n", ref_dot);
    printf("  leg A rot dot  %+.6f  rel-err %.4f  %s\n", dot_rot,   a_err, a_err  < 1e-3f ? "PASS" : "FAIL");
    printf("  leg B quant dot%+.6f  rel-err %.4f  %s (3-bit noise budget 0.15)\n", dot_quant, b_err, b_err < 0.15f ? "PASS" : "FAIL");
    printf("  leg C quant NMSE %.5f            %s (budget 0.06)\n", c_nmse, c_nmse < 0.06f ? "PASS" : "FAIL");

    int fails = (a_err >= 1e-3f) + (b_err >= 0.15f) + (c_nmse >= 0.06f);
    ggml_free(ctx);
    return fails;
}

int main(void) {
    int fails = 0;
    // control first: this leg passing is what validates the probe itself
    fails += probe(128, 7);
    fails += probe(64, 11);   // gpt-oss shape
    fails += probe(576, 13);  // deepseek4 MLA shape
    printf(fails ? "PROBE: %d leg(s) FAILED\n" : "PROBE: all legs pass\n", fails);
    return fails ? 1 : 0;
}
