// probe-pad-k-chain2: statistically powered version, plus a vec_dot leg.
//
// v1 found: rotations agree (A exact, C at quant noise) but single-pair
// quantized dots blew out on padded shapes only. Two instrument gaps: one
// pair has no statistical power at these dims, and v1 hand-computed dots in
// f32, bypassing the ggml vec_dot path that real FA consumption uses.
//
// v2 metrics, per shape (128 native control / 64->128 gpt-oss / 576->640 dsv4):
//   AMP  = RMS dot error of quantized-K dots, normalized per pair by
//          ||Qr||*||Kr||, over NQ*NK pairs — reported as ratio vs the native
//          control so the number is "how much worse does padding make it".
//   VDOT = same but the dot computed by ggml_mul_mat(cache, Qr) — the actual
//          CPU vec_dot consumption path. VDOT >> AMP convicts vec_dot.
//   PADE = post-inverse-rotation energy in the pad region of dequantized K,
//          as fraction of real-region energy (zero-structure preservation).
#include "ggml.h"
#include "ggml-cpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { NK = 1024, NQ = 32 };

static double rms_dot_err(const float * Q, const float * Kq, const float * Kr,
                          int pd, int nq, int nk) {
    double acc = 0; long n = 0;
    for (int iq = 0; iq < nq; iq++) {
        const float * qv = Q + (size_t) iq * pd;
        double qn = 0; for (int i = 0; i < pd; i++) qn += (double) qv[i]*qv[i];
        qn = sqrt(qn);
        for (int ik = 0; ik < nk; ik++) {
            const float * kq = Kq + (size_t) ik * pd;
            const float * kr = Kr + (size_t) ik * pd;
            double dq = 0, dr = 0, kn = 0;
            for (int i = 0; i < pd; i++) {
                dq += (double) qv[i]*kq[i];
                dr += (double) qv[i]*kr[i];
                kn += (double) kr[i]*kr[i];
            }
            kn = sqrt(kn);
            double e = (dq - dr) / (qn * kn > 0 ? qn * kn : 1);
            acc += e*e; n++;
        }
    }
    return sqrt(acc / n);
}

static double probe(int d, unsigned seed, double * vdot_out, double * pade_out) {
    const int pd = ((d + 127) / 128) * 128;
    struct ggml_init_params ip = { .mem_size = 1536u*1024*1024, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, NQ);
    struct ggml_tensor * k = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, NK);
    srand(seed);
    for (int64_t i = 0; i < ggml_nelements(q); i++) ((float *) q->data)[i] = (float) rand()/RAND_MAX*2 - 1;
    for (int64_t i = 0; i < ggml_nelements(k); i++) ((float *) k->data)[i] = (float) rand()/RAND_MAX*2 - 1;

    struct ggml_tensor * qp = (d != pd) ? ggml_cont(ctx, ggml_pad(ctx, q, pd - d, 0, 0, 0)) : q;
    struct ggml_tensor * kp = (d != pd) ? ggml_cont(ctx, ggml_pad(ctx, k, pd - d, 0, 0, 0)) : k;
    struct ggml_tensor * q_rot = ggml_turbo_wht(ctx, qp, 0, 0, NULL);
    struct ggml_tensor * k_rot = ggml_turbo_wht(ctx, kp, 0, 0, NULL);

    struct ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_TURBO3_0, pd, NK);
    struct ggml_tensor * idx   = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, NK);
    for (int i = 0; i < NK; i++) ((int64_t *) idx->data)[i] = i;
    struct ggml_tensor * wr = ggml_set_rows(ctx, cache, kp, idx);
    { int32_t g = 128; memcpy(wr->op_params, &g, sizeof g); }

    struct ggml_tensor * back = ggml_cpy(ctx, cache, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, pd, NK));
    // the real consumption path: mul_mat(turbo3 cache, rotated Q) -> vec_dot
    struct ggml_tensor * vd   = ggml_mul_mat(ctx, cache, q_rot);
    // zero-structure: rotate the dequantized rows back and look at the pad region
    struct ggml_tensor * kinv = ggml_turbo_wht(ctx, back, 1, 0, NULL);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf, q_rot);
    ggml_build_forward_expand(gf, k_rot);
    ggml_build_forward_expand(gf, wr);
    ggml_build_forward_expand(gf, back);
    ggml_build_forward_expand(gf, vd);
    ggml_build_forward_expand(gf, kinv);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    const float * Qr = (const float *) q_rot->data;
    const float * Kr = (const float *) k_rot->data;
    const float * Kq = (const float *) back->data;

    double amp = rms_dot_err(Qr, Kq, Kr, pd, NQ, NK);

    // VDOT: vd is (NK, NQ): vd[iq*NK + ik] = dot(cache row ik, Qr row iq)
    {
        double acc = 0; long n = 0;
        const float * V = (const float *) vd->data;
        for (int iq = 0; iq < NQ; iq++) {
            const float * qv = Qr + (size_t) iq * pd;
            double qn = 0; for (int i = 0; i < pd; i++) qn += (double) qv[i]*qv[i];
            qn = sqrt(qn);
            for (int ik = 0; ik < NK; ik++) {
                const float * kr = Kr + (size_t) ik * pd;
                double dr = 0, kn = 0;
                for (int i = 0; i < pd; i++) { dr += (double) qv[i]*kr[i]; kn += (double) kr[i]*kr[i]; }
                kn = sqrt(kn);
                double e = (V[(size_t) iq * NK + ik] - dr) / (qn * kn > 0 ? qn * kn : 1);
                acc += e*e; n++;
            }
        }
        *vdot_out = sqrt(acc / n);
    }

    // PADE: energy in [d:pd] vs [0:d] of inverse-rotated dequantized K
    if (d != pd) {
        const float * Ki = (const float *) kinv->data;
        double real_e = 0, pad_e = 0;
        for (int ik = 0; ik < NK; ik++) {
            const float * r = Ki + (size_t) ik * pd;
            for (int i = 0; i < d;  i++) real_e += (double) r[i]*r[i];
            for (int i = d; i < pd; i++) pad_e  += (double) r[i]*r[i];
        }
        *pade_out = pad_e / (real_e > 0 ? real_e : 1);
    } else {
        *pade_out = 0;
    }

    ggml_free(ctx);
    return amp;
}

int main(void) {
    double vd, pe;
    double base = probe(128, 7, &vd, &pe);
    printf("native 128 : AMP %.5f  VDOT %.5f  PADE %.4f  (control)\n", base, vd, pe);
    double a64 = probe(64, 11, &vd, &pe);
    printf("64 -> 128  : AMP %.5f (%.2fx)  VDOT %.5f  PADE %.4f\n", a64, a64/base, vd, pe);
    double a576 = probe(576, 13, &vd, &pe);
    printf("576 -> 640 : AMP %.5f (%.2fx)  VDOT %.5f  PADE %.4f\n", a576, a576/base, vd, pe);
    return 0;
}
