// Tensor-level check of the HEPH glue fusions: build each chain as a small graph, run it on the GPU backend
// (where the fusion fires) and on the CPU backend (reference), compare outputs.
// Usage: fusetest [case-substring]   Exit code 1 on any mismatch.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

extern "C" int ggml_cuda_heph_fuse_count;

struct built {
    std::vector<ggml_tensor *> inputs;
    std::vector<ggml_tensor *> outputs;
    std::vector<ggml_tensor *> silent; // members of outputs that are only placed in the graph at that point, not read back
};
using builder = std::function<built(ggml_context *)>;

static ggml_tensor * inp(ggml_context * ctx, built & b, int64_t n0, int64_t n1 = 1, int64_t n2 = 1) {
    ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n0, n1, n2);
    ggml_set_input(t);
    b.inputs.push_back(t);
    return t;
}

static double g_bench_us = 0.0;
static std::string g_case; // name of the case being run
static size_t g_repeat_bad = 0; // FUSEREPEAT: output reads that differed from the first evaluation
static bool g_test_side = false; // true while building the graph under test, false for the reference graph
static std::vector<bool> g_out_f16; // which outputs of the last run were F16
static std::vector<bool> g_out_t3;  // which were turbo3 (compared after decoding)

static std::vector<std::vector<float>> run(ggml_backend_t backend, const builder & build, unsigned seed) {
    ggml_init_params ip = { ggml_tensor_overhead()*4096 + ggml_graph_overhead_custom(4096, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    built b = build(ctx);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    const auto is_silent = [&](const ggml_tensor * o) { return std::find(b.silent.begin(), b.silent.end(), o) != b.silent.end(); };
    for (ggml_tensor * o : b.outputs) {
        if (!is_silent(o)) {
            ggml_set_output(o);
        }
        ggml_build_forward_expand(gf, o);
    }
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(ga, gf)) { fprintf(stderr, "alloc failed\n"); exit(2); }
    // Two evaluations of the same graph with different inputs: anything a backend carries over between
    // evaluations (cached quantized inputs, captured device graphs) has to follow the new data.
    const auto set_inputs = [&](unsigned pass) {
    std::mt19937 rng(seed + 1000*pass);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (ggml_tensor * t : b.inputs) {
        if (t->type == GGML_TYPE_I32) { // relative distances in [0, extent]
            std::vector<int32_t> v(ggml_nelements(t));
            for (int32_t & f : v) f = rng() % 1025;
            ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
        } else if (t->type == GGML_TYPE_I64) { // distinct cache rows, one per token
            std::vector<int64_t> v(ggml_nelements(t));
            const bool all = strncmp(t->name, "rows_all", 8) == 0; // every cache row in order, or scattered rows
            for (size_t i = 0; i < v.size(); ++i) v[i] = all ? (int64_t) i : (int64_t) (3*i + 1 + rng() % 2);
            ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
        } else if (t->type == GGML_TYPE_TURBO3_0) { // an empty KV cache
            std::vector<uint8_t> v(ggml_nbytes(t), 0);
            ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
        } else if (t->type == GGML_TYPE_Q8_0) { // a quantized weight
            std::vector<float> v(ggml_nelements(t));
            for (float & f : v) f = nd(rng);
            std::vector<uint8_t> q(ggml_nbytes(t));
            ggml_quantize_chunk(t->type, v.data(), q.data(), 0, ggml_nelements(t)/t->ne[0], t->ne[0], nullptr);
            ggml_backend_tensor_set(t, q.data(), 0, ggml_nbytes(t));
        } else if (t->type == GGML_TYPE_F16) { // an attention mask: 0 or -inf
            std::vector<ggml_fp16_t> v(ggml_nelements(t));
            // "mask_blocks": also about a third of the runs of 64 keys masked for all heads (no period, so that every
            // block layout meets masked runs before and after unmasked ones): whole waves of a kernel see no key
            const bool blocks = strncmp(t->name, "mask_blocks", 11) == 0;
            for (size_t e = 0; e < v.size(); ++e) {
                const uint32_t run64 = (uint32_t) ((e % t->ne[0])/64);
                const uint32_t kind  = blocks ? ((run64*2654435761u) >> 24) % 3 : 0;
                // kind 2: a bias of -12, keys whose weight is tiny but not zero. Skipping those would be an error
                // that the CPU comparison is too loose to see and a bit compare against the no-skip kernel does see.
                v[e] = ggml_fp32_to_fp16(kind == 2 ? -12.0f : kind == 1 || rng() % 4 == 0 ? -INFINITY : 0.0f);
            }
            ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
        } else {
            std::vector<float> v(ggml_nelements(t));
            for (float & f : v) f = nd(rng);
            ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
        }
    }
    };
    for (unsigned pass = 0; pass < 2; ++pass) {
    set_inputs(pass);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(2); }
    }
    if (const char * nb = getenv("FUSEBENCH")) { // time repeated evaluations of the whole graph, launch overhead included
        const int n = atoi(nb);
        ggml_backend_synchronize(backend);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n; ++i) ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        g_bench_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count()/n;
    }
    std::vector<std::vector<float>> res;
    g_out_f16.clear();
    g_out_t3.clear();
    for (ggml_tensor * o : b.outputs) {
        if (is_silent(o)) {
            continue;
        }
        std::vector<float> v(ggml_nelements(o));
        g_out_f16.push_back(o->type == GGML_TYPE_F16);
        g_out_t3.push_back(o->type == GGML_TYPE_TURBO3_0);
        if (o->type == GGML_TYPE_TURBO3_0) { // decode: norm times the centroid of each 3-bit code
            static const float centroids[8] = { -0.190685f, -0.117832f, -0.065717f, -0.021460f, 0.021460f, 0.065717f, 0.117832f, 0.190685f };
            std::vector<uint8_t> raw(ggml_nbytes(o));
            ggml_backend_tensor_get(o, raw.data(), 0, raw.size());
            for (size_t blk = 0; blk < raw.size()/50; ++blk) {
                const uint8_t * p = raw.data() + 50*blk;
                ggml_fp16_t h; memcpy(&h, p, 2);
                const float norm = ggml_fp16_to_fp32(h);
                for (int i = 0; i < 128; ++i) {
                    const int low = (p[2 + i/4] >> (2*(i % 4))) & 3;
                    const int high = (p[2 + 32 + i/8] >> (i % 8)) & 1;
                    v[128*blk + i] = norm*centroids[low | (high << 2)];
                }
            }
        } else if (o->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> h(ggml_nelements(o));
            ggml_backend_tensor_get(o, h.data(), 0, ggml_nbytes(o));
            for (size_t i = 0; i < h.size(); ++i) v[i] = ggml_fp16_to_fp32(h[i]);
        } else {
            ggml_backend_tensor_get(o, v.data(), 0, ggml_nbytes(o));
        }
        res.push_back(v);
    }
    if (const char * dir = g_test_side ? getenv("FUSEDUMP") : nullptr) { // raw outputs, to compare two builds bit for bit
        for (size_t k = 0; k < res.size(); ++k) {
            const std::string path = std::string(dir) + "/" + g_case + "-s" + std::to_string(seed) + "-o" + std::to_string(k) + ".bin";
            FILE * fp = fopen(path.c_str(), "wb");
            if (fp) { fwrite(res[k].data(), sizeof(float), res[k].size(), fp); fclose(fp); }
        }
    }
    if (const char * nr = g_test_side ? getenv("FUSEREPEAT") : nullptr) { // determinism: the same graph and inputs must give the same bytes every time
        size_t bad = 0;
        const int n = atoi(nr);
        for (int i = 0; i < n; ++i) {
            set_inputs(1); // ops may work in place on what they read
            ggml_backend_graph_compute(backend, gf);
            size_t k = 0;
            for (ggml_tensor * o : b.outputs) {
                if (is_silent(o)) continue;
                if (o->type == GGML_TYPE_F32) {
                    std::vector<float> v(ggml_nelements(o));
                    ggml_backend_tensor_get(o, v.data(), 0, ggml_nbytes(o));
                    bad += memcmp(v.data(), res[k].data(), ggml_nbytes(o)) != 0;
                }
                ++k;
            }
        }
        if (bad) fprintf(stderr, "  REPEAT MISMATCH: %zu output reads differed over %d evaluations\n", bad, n);
        g_repeat_bad += bad;
    }
    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return res;
}

// cpu_both: the graph under test also runs on the CPU backend (checks a CPU implementation against the reference chain)
// tol: NMSE bound against the CPU reference. plain: a stock kernel is under test, no fusion has to fire.
struct test_case { std::string name; builder build; bool cpu_both = false; double tol = 1e-9; bool plain = false; };

// --- the chains, written exactly as src/models/inkling2.cpp writes them ---
static ggml_tensor * clip(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * scale, float threshold) {
    if (scale) cur = ggml_mul(ctx, cur, scale);
    ggml_tensor * norm = ggml_sqrt(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, cur)));
    norm = ggml_scale_bias(ctx, norm, 1.0f, 1e-6f);
    ggml_tensor * factor = ggml_div(ctx, ggml_fill(ctx, norm, threshold), norm);
    return ggml_mul(ctx, cur, ggml_clamp(ctx, factor, 0.0f, 1.0f));
}


static ggml_tensor * mix(ggml_context * ctx, const std::vector<ggml_tensor *> & blocks, ggml_tensor * cur, ggml_tensor * query, int64_t n_embd, int64_t n_tokens) {
    ggml_tensor * states = ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens);
    for (ggml_tensor * block : blocks) {
        states = ggml_concat(ctx, ggml_reshape_3d(ctx, block, n_embd, 1, n_tokens), states, 1);
    }
    states = ggml_rms_norm(ctx, states, 1e-6f);
    ggml_tensor * queries = ggml_repeat(ctx, query, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, query->ne[1], n_tokens));
    ggml_tensor * scores = ggml_mul_mat(ctx, states, queries);
    scores = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f/sqrtf(n_embd), 0.0f);
    return ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, states)), scores);
}

static ggml_tensor * control(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * message, ggml_tensor * down, ggml_tensor * wgain, ggml_tensor * wforget) {
    ggml_tensor * gate = ggml_concat(ctx, ggml_rms_norm(ctx, cur, 1e-6f), ggml_rms_norm(ctx, message, 1e-6f), 0);
    gate = ggml_mul_mat(ctx, down, gate);
    ggml_tensor * gain   = ggml_tanh(ctx, ggml_mul_mat(ctx, wgain,   gate));
    ggml_tensor * forget = ggml_tanh(ctx, ggml_mul_mat(ctx, wforget, gate));
    return ggml_add(ctx, ggml_sub(ctx, cur, ggml_mul(ctx, forget, cur)), ggml_mul(ctx, gain, message));
}

static ggml_tensor * situ(ggml_context * ctx, ggml_tensor * gate_up, int64_t n_ff, float beta, float lb) {
    ggml_tensor * cur = ggml_view_3d(ctx, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
    ggml_tensor * up  = ggml_view_3d(ctx, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);
    ggml_tensor * act = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, cur, 1.0f/beta)), beta);
    act = ggml_mul(ctx, act, ggml_sigmoid(ctx, cur));
    up = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, up, 1.0f/lb)), lb);
    return ggml_mul(ctx, act, up);
}

#include "cases.inc"

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!gpu || !cpu) { fprintf(stderr, "backend init failed\n"); return 2; }
    int fails = 0;
    for (const test_case & tc : cases()) {
        if (argc > 1 && tc.name.find(argv[1]) == std::string::npos) continue;
        double worst = 0.0;
        g_case = tc.name;
        const int fused_before = ggml_cuda_heph_fuse_count; // a reused device graph does not re-run the matcher, so count per case
        for (unsigned seed = 1; seed <= 3; ++seed) {
            g_test_side = true;
            auto g = run(tc.cpu_both ? cpu : gpu, tc.build, seed);
            g_test_side = false;
            const double gpu_us = g_bench_us;
            auto c = run(cpu, tc.build, seed);
            if (seed == 3 && getenv("FUSEBENCH")) printf("  bench %-28s %.1f us/graph\n", tc.name.c_str(), gpu_us);
            for (size_t o = 0; o < g.size(); ++o) {
                double num = 0.0, den = 0.0;
                if (g_out_t3[o]) {
                    // 3-bit codes: an input that differs in the last float bit can land on the other side of a level
                    // threshold, which moves that element by a whole level. Allow that for at most 0.2 % of the elements;
                    // a packing or indexing error moves far more.
                    size_t differ = 0;
                    for (size_t i = 0; i < g[o].size(); ++i) differ += std::fabs((double) g[o][i] - c[o][i]) > 1e-3*std::fabs(c[o][i]) + 1e-12;
                    if (differ > g[o].size()/500) worst = INFINITY;
                    if (differ) fprintf(stderr, "  %s: %zu of %zu turbo3 elements differ\n", tc.name.c_str(), differ, g[o].size());
                    continue;
                }
                for (size_t i = 0; i < g[o].size(); ++i) {
                    if (std::isinf(c[o][i]) && g[o][i] == c[o][i]) continue; // the same infinity on both sides
                    const double d = (double) g[o][i] - c[o][i];
                    num += d*d; den += (double) c[o][i]*c[o][i];
                    if (!std::isfinite(g[o][i]) || !std::isfinite(c[o][i])) num = INFINITY;
                }
                // F16 outputs: a last-bit float difference can flip a whole half step (1e-3 relative), measured up to 2e-9.
                // A 0.1 % kernel error gives 1e-6 either way.
                const double nm = num/(den + 1e-30)*(g_out_f16[o] ? 0.1 : 1.0);
                if (!(nm < tc.tol)) { // say where: which output, and the first element that is off
                    size_t first = 0;
                    while (first < g[o].size() && std::fabs((double) g[o][first] - c[o][first]) <= 1e-4*std::fabs(c[o][first]) + 1e-9) ++first;
                    fprintf(stderr, "  %s: seed %u output %zu of %zu, nmse %.3g, %zu elements, first off at %zu: test %g ref %g\n", tc.name.c_str(), seed, o, g.size(),
                            nm, g[o].size(), first, first < g[o].size() ? g[o][first] : 0.0f, first < c[o].size() ? c[o][first] : 0.0f);
                }
                worst = std::max(worst, nm);
            }
        }
        if (!tc.cpu_both && !tc.plain && ggml_cuda_heph_fuse_count == fused_before) { worst = INFINITY; fprintf(stderr, "%s: fusion did not fire\n", tc.name.c_str()); }
        if (g_repeat_bad) { worst = INFINITY; g_repeat_bad = 0; }
        const bool ok = worst < tc.tol;
        printf("%s %-28s nmse=%.3g\n", ok ? "PASS" : "FAIL", tc.name.c_str(), worst);
        fails += !ok;
    }
    return fails ? 1 : 0;
}
