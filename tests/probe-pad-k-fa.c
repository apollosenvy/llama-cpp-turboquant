// probe-pad-k-fa: integrated replication of the exact layer-0 geometry from the
// gpt-oss eval-callback dump, CPU backend. Components all pass in isolation
// (probe-pad-k-chain2); this one adds what those probes lacked: the real cache
// geometry (512-row turbo3 cache, 8 heads merged per row), the 3D/4D
// view+permute sequence, and FLASH_ATTN_EXT itself.
//
//   Q {64,64,4} -> pad {128,64,4} -> turbo_wht -> permute {128,4,64}
//   K {64, 8,4} -> pad {128,8,4} -> view {1024,4} -> set_rows(turbo3 cache {1024,512})
//                  cache view {128,8,256} -> permute {128,256,8}
//   V {64, 8,4} -> f16 cache {512,512} path, view {64,8,256} -> permute {64,256,8}
//   FLASH_ATTN_EXT(q,k,v,mask,scale=1/8)  vs  same with K kept f16.
//
// Verdict metric: NMSE between turbo3-K FA output and f16-K FA output.
// Expected if healthy: ~3-bit quant noise (<0.1). The KL gate says the real
// model behaves as if this were ~O(1).
#include "ggml.h"
#include "ggml-cpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { D = 64, PD = 128, NH = 64, NKV = 8, T = 4, CACHE = 256 };

static struct ggml_tensor * build_fa(struct ggml_context * ctx, int use_turbo,
        struct ggml_tensor * q, struct ggml_tensor * k, struct ggml_tensor * v,
        struct ggml_tensor * mask, struct ggml_tensor * idx, struct ggml_tensor * sinks) {
    // Q side
    struct ggml_tensor * qq = q;
    if (use_turbo) {
        qq = ggml_cont(ctx, ggml_pad(ctx, qq, PD - D, 0, 0, 0));
        qq = ggml_turbo_wht(ctx, qq, 0, 0, NULL);
    }
    qq = ggml_permute(ctx, qq, 0, 2, 1, 3); // (Dq, T, NH)

    // K side: write into a cache like cpy_k does, read back like get_k does
    struct ggml_tensor * kread;
    if (use_turbo) {
        struct ggml_tensor * kp = ggml_cont(ctx, ggml_pad(ctx, k, PD - D, 0, 0, 0)); // {128,8,T}
        struct ggml_tensor * kmerged = ggml_view_2d(ctx, kp, PD*NKV, T, kp->nb[2], 0);
        struct ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_TURBO3_0, PD*NKV, CACHE);
        struct ggml_tensor * wr = ggml_set_rows(ctx, cache, kmerged, idx);
        { int32_t g = 128; memcpy(wr->op_params, &g, sizeof g); }
        kread = ggml_view_3d(ctx, cache, PD, NKV, CACHE,
                             ggml_row_size(cache->type, PD),
                             ggml_row_size(cache->type, PD*NKV), 0);
        kread->src[0] = wr; // force dependency so set_rows runs first
    } else {
        struct ggml_tensor * kf = ggml_cast(ctx, k, GGML_TYPE_F16); // {64,8,T}
        struct ggml_tensor * kmerged = ggml_view_2d(ctx, kf, D*NKV, T, kf->nb[2], 0);
        struct ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, D*NKV, CACHE);
        struct ggml_tensor * wr = ggml_set_rows(ctx, cache, kmerged, idx);
        kread = ggml_view_3d(ctx, cache, D, NKV, CACHE,
                             ggml_row_size(cache->type, D),
                             ggml_row_size(cache->type, D*NKV), 0);
        kread->src[0] = wr;
    }
    kread = ggml_permute(ctx, kread, 0, 2, 1, 3); // (Dk, CACHE, NKV)

    // V side: f16 cache both legs (isolate K)
    struct ggml_tensor * vf = ggml_cast(ctx, v, GGML_TYPE_F16);
    struct ggml_tensor * vmerged = ggml_view_2d(ctx, vf, D*NKV, T, vf->nb[2], 0);
    struct ggml_tensor * vcache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, D*NKV, CACHE);
    struct ggml_tensor * vwr = ggml_set_rows(ctx, vcache, vmerged, idx);
    struct ggml_tensor * vread = ggml_view_3d(ctx, vcache, D, NKV, CACHE,
                         ggml_row_size(vcache->type, D),
                         ggml_row_size(vcache->type, D*NKV), 0);
    vread->src[0] = vwr;
    vread = ggml_permute(ctx, vread, 0, 2, 1, 3);

    if (getenv("PROBE_DIAG") && use_turbo) {
        // stash rotated-K reference and dequantized cache for main() diagnostics
        extern struct ggml_tensor * g_krot, * g_kback, * g_qrot;
        struct ggml_tensor * kp2 = ggml_cont(ctx, ggml_pad(ctx, k, PD - D, 0, 0, 0));
        g_krot  = ggml_turbo_wht(ctx, kp2, 0, 0, NULL);
        g_kback = ggml_cpy(ctx, kread->view_src ? kread->view_src : kread,
                           ggml_new_tensor_2d(ctx, GGML_TYPE_F32, PD*NKV, CACHE));
        g_qrot  = qq;
    }
    struct ggml_tensor * out = ggml_flash_attn_ext(ctx, qq, kread, vread, mask,
                                                   1.0f/sqrtf((float) D), 0.0f, 0.0f);
    if (sinks) ggml_flash_attn_ext_add_sinks(out, sinks);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    return out;
}

struct ggml_tensor * g_krot, * g_kback, * g_qrot;

int main(void) {
    struct ggml_init_params ip = { .mem_size = 1024u*1024*1024, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, NH, T);
    struct ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, NKV, T);
    struct ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, NKV, T);
    srand(42);
    for (int64_t i = 0; i < ggml_nelements(q); i++) ((float *) q->data)[i] = (float) rand()/RAND_MAX*2 - 1;
    for (int64_t i = 0; i < ggml_nelements(k); i++) ((float *) k->data)[i] = (float) rand()/RAND_MAX*2 - 1;
    // PROBE_LOAD_DIR: replace synthetic Q/K/V with real layer-0 activations
    // dumped by the COMMON_DEBUG_DUMP_DIR patch (i64 ne[4] header + raw f32).
    if (getenv("PROBE_LOAD_DIR")) {
        const char * dir = getenv("PROBE_LOAD_DIR");
        const char * names[3] = { "Qcur-0.3.bin", "Kcur-0.3.bin", "Vcur-0.2.bin" };
        struct ggml_tensor * dsts[3]; dsts[0] = q; dsts[1] = k; dsts[2] = v;
        for (int f = 0; f < 3; f++) {
            char path[512]; snprintf(path, sizeof path, "%s/%s", dir, names[f]);
            FILE * fp = fopen(path, "rb");
            if (!fp) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
            int64_t ne[4]; if (fread(ne, sizeof ne, 1, fp) != 1) exit(2);
            if (ne[0] != dsts[f]->ne[0] || ne[1] != dsts[f]->ne[1] || ne[2] != dsts[f]->ne[2]) {
                fprintf(stderr, "%s shape {%ld,%ld,%ld} != probe {%ld,%ld,%ld}\n", names[f],
                        ne[0], ne[1], ne[2], dsts[f]->ne[0], dsts[f]->ne[1], dsts[f]->ne[2]);
                exit(2);
            }
            if (fread(dsts[f]->data, 1, ggml_nbytes(dsts[f]), fp) != ggml_nbytes(dsts[f])) exit(2);
            fclose(fp);
        }
        fprintf(stderr, "loaded real layer-0 activations from %s\n", dir);
    }
    // OUTLIER_INJECT: mimic gpt-oss sink-bias K structure — same dims hot across
    // all rows (bias-like), magnitudes like the eval dump (|K| up to ~23)
    if (getenv("PROBE_K_OUTLIER")) {
        float s = atof(getenv("PROBE_K_OUTLIER"));
        for (int t = 0; t < T; t++) for (int h = 0; h < NKV; h++) {
            float * row = (float *) k->data + ((size_t) t*NKV + h) * D;
            row[0] *= s; row[7] *= s * 0.5f; row[19] *= s * 0.25f;
        }
    }
    for (int64_t i = 0; i < ggml_nelements(v); i++) ((float *) v->data)[i] = (float) rand()/RAND_MAX*2 - 1;

    // causal mask over the cache: token t sees cache rows 0..t, rest -inf
    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, CACHE, GGML_PAD(T, 64));
    {
        ggml_fp16_t * m = (ggml_fp16_t *) mask->data;
        ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY), zero = ggml_fp32_to_fp16(0.0f);
        for (int64_t r = 0; r < mask->ne[1]; r++)
            for (int64_t c = 0; c < CACHE; c++)
                m[r*CACHE + c] = (r < T && c <= r) ? zero : ninf;
    }
    struct ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, T);
    for (int i = 0; i < T; i++) ((int64_t *) idx->data)[i] = i;

    struct ggml_tensor * sinks = NULL;
    if (getenv("PROBE_SINKS")) {
        sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, NH);
        for (int i = 0; i < NH; i++) ((float *) sinks->data)[i] = 0.5f + 3.0f * i / NH;
    }
    struct ggml_tensor * out_ref = build_fa(ctx, 0, q, k, v, mask, idx, sinks);
    struct ggml_tensor * out_t3  = build_fa(ctx, 1, q, k, v, mask, idx, sinks);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(gf, out_ref);
    ggml_build_forward_expand(gf, out_t3);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    const float * A = (const float *) out_t3->data;
    const float * R = (const float *) out_ref->data;
    double e = 0, rr = 0;
    for (int64_t i = 0; i < ggml_nelements(out_ref); i++) { double d = A[i]-R[i]; e += d*d; rr += (double) R[i]*R[i]; }
    double nmse = e / (rr > 0 ? rr : 1);
    printf("integrated FA turbo3-K vs f16-K NMSE = %.5f  -> %s\n", nmse,
           nmse < 0.1 ? "HEALTHY (bug is elsewhere)" : "REPRODUCED (bug is in this graph)");
    printf("sample out ref: %+.4f %+.4f %+.4f %+.4f\n", R[0], R[1], R[2], R[3]);
    printf("sample out t3 : %+.4f %+.4f %+.4f %+.4f\n", A[0], A[1], A[2], A[3]);
    if (getenv("PROBE_DIAG")) {
        ggml_build_forward_expand(gf, g_krot);
        ggml_build_forward_expand(gf, g_kback);
        ggml_graph_compute_with_ctx(ctx, gf, 4);
        // per (token,head) row: cos(decode, rotated-ref) and norm ratio
        for (int t = 0; t < T; t++) for (int h = 0; h < 2; h++) {
            const float * kr = (const float *) g_krot->data + ((size_t) t*NKV + h) * PD;
            const float * kb = (const float *) g_kback->data + (size_t) t * (PD*NKV) + (size_t) h * PD;
            double d = 0, n1 = 0, n2 = 0;
            for (int i = 0; i < PD; i++) { d += (double) kr[i]*kb[i]; n1 += (double) kr[i]*kr[i]; n2 += (double) kb[i]*kb[i]; }
            printf("  t%d h%d: cos(decode,rot-ref)=%+.4f  |decode|/|ref|=%.4f\n",
                   t, h, d/sqrt(n1*n2), sqrt(n2/n1));
        }
        // logits: token 3 attends rows 0..3, head 0 (q head 0 -> kv head 0)
        const float * qr = (const float *) ggml_get_data(g_qrot); // {PD, T, NH} permuted view base
        // qq is a permuted view of the wht output {PD, NH, T}; index directly on the source
        struct ggml_tensor * qsrc = g_qrot->view_src ? g_qrot->view_src : g_qrot;
        const float * qs = (const float *) qsrc->data; // {PD, NH, T}
        for (int c = 0; c < T; c++) {
            const float * qv = qs + ((size_t) 3 * 64 + 0) * PD; // token 3, q-head 0
            const float * kb = (const float *) g_kback->data + (size_t) c * (PD*NKV) + 0; // row c, kv-head 0
            double lt = 0; for (int i = 0; i < PD; i++) lt += (double) qv[i]*kb[i];
            // reference logit from ORIGINAL unrotated q/k, and graph-rot logit
            const float * q0 = (const float *) q->data + ((size_t) 3 * NH + 0) * D;
            const float * k0 = (const float *) k->data + ((size_t) c * NKV + 0) * D;
            double lr = 0; for (int i = 0; i < D; i++) lr += (double) q0[i]*k0[i];
            const float * krr = (const float *) g_krot->data + ((size_t) c*NKV + 0) * PD;
            double lg = 0; for (int i = 0; i < PD; i++) lg += (double) qv[i]*krr[i];
            printf("  logit t3h0 <- row %d: ref=%.3f  graphrot=%.3f  turbo=%.3f\n",
                   c, lr/8.0, lg/8.0, lt/8.0);
        }
        (void) qr;
    }
    ggml_free(ctx);
    return nmse >= 0.1;
}
