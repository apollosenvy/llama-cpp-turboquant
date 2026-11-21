static std::vector<test_case> cases() {
    std::vector<test_case> v;
    for (int T : {1, 5}) {
        v.push_back({"clip_q_T" + std::to_string(T), [T](ggml_context * ctx) {
            built b; ggml_tensor * x = inp(ctx, b, 128, 24, T); ggml_tensor * s = inp(ctx, b, 128, 24);
            // feed x through an op first so it is a computed node like in the model
            b.outputs = { clip(ctx, ggml_scale(ctx, x, 9.0f), s, 22.627417f) }; return b; }});
        v.push_back({"clip_k_T" + std::to_string(T), [T](ggml_context * ctx) {
            built b; ggml_tensor * x = inp(ctx, b, 128, 8, T);
            b.outputs = { clip(ctx, ggml_scale(ctx, x, 3.0f), nullptr, 22.627417f) }; return b; }});
        v.push_back({"clip_r_T" + std::to_string(T), [T](ggml_context * ctx) {
            built b; ggml_tensor * x = inp(ctx, b, 16, 24, T); ggml_tensor * s = inp(ctx, b, 16, 24);
            b.outputs = { clip(ctx, ggml_scale(ctx, x, 4.0f), s, 8.0f) }; return b; }});
    }
    for (int T : {1, 5}) for (int nq : {1, 3, 4}) for (int S : {1, 2, 3, 5}) {
        v.push_back({"mix_T" + std::to_string(T) + "_q" + std::to_string(nq) + "_S" + std::to_string(S), [T, nq, S](ggml_context * ctx) {
            built b; const int64_t n = 2304;
            std::vector<ggml_tensor *> blocks;
            for (int k = 0; k < S - 1; ++k) blocks.push_back(ggml_scale(ctx, inp(ctx, b, n, T), 0.5f + k));
            ggml_tensor * cur = ggml_scale(ctx, inp(ctx, b, n, T), 2.0f);
            ggml_tensor * q = inp(ctx, b, n, nq);
            // scale the query up so the softmax is far from uniform
            ggml_tensor * out = mix(ctx, blocks, cur, q, n, T);
            // in the model the query is a weight; keep it alive past the mix so the allocator cannot hand its memory to out
            b.outputs = { out, ggml_scale(ctx, q, 1.0f) }; return b; }});
    }
    for (int T : {1, 5}) {
        v.push_back({"control_T" + std::to_string(T), [T](ggml_context * ctx) {
            built b; const int64_t n = 2304, r = 64;
            ggml_tensor * cur = ggml_scale(ctx, inp(ctx, b, n, T), 1.5f);
            ggml_tensor * msg = ggml_scale(ctx, inp(ctx, b, n, T), 0.7f);
            ggml_tensor * down = inp(ctx, b, 2*n, r); ggml_tensor * wg = inp(ctx, b, r, n); ggml_tensor * wf = inp(ctx, b, r, n);
            // weights at a model-like scale so tanh is not saturated
            ggml_tensor * out = control(ctx, cur, msg, ggml_scale(ctx, down, 0.02f), wg, wf);
            b.outputs = { ggml_scale(ctx, out, 1.0f), ggml_scale(ctx, wg, 1.0f), ggml_scale(ctx, wf, 1.0f) }; return b; }});
        v.push_back({"situ_T" + std::to_string(T), [T](ggml_context * ctx) {
            built b; const int64_t n_ff = 1152;
            ggml_tensor * gu = ggml_scale(ctx, inp(ctx, b, 2*n_ff, 8, T), 3.0f);
            b.outputs = { situ(ctx, gu, n_ff, 4.0f, 25.0f) }; return b; }});
        for (int C : {1024, 2304}) {
            v.push_back({"conv_T" + std::to_string(T) + "_C" + std::to_string(C), [T, C](ggml_context * ctx) {
                built b; const int64_t n_conv = 4;
                ggml_tensor * x = ggml_scale(ctx, inp(ctx, b, n_conv - 1 + T, C, 1), 1.0f);
                ggml_tensor * w = inp(ctx, b, n_conv, C);
                ggml_tensor * cur = ggml_scale(ctx, inp(ctx, b, C, T), 2.0f);
                ggml_tensor * out = ggml_reshape_2d(ctx, ggml_ssm_conv(ctx, x, w), C, T);
                out = ggml_scale(ctx, ggml_add(ctx, cur, out), 1.0f/sqrtf(n_conv + 1));
                b.outputs = { out, ggml_scale(ctx, w, 1.0f) }; return b; }});
        }
    }
    for (int T : {1, 5}) {
        v.push_back({"controlraw_T" + std::to_string(T), [T](ggml_context * ctx) {
            built b; const int64_t n = 2304, r = 64;
            // cur and msg are columns of mix outputs, as in build_layer
            ggml_tensor * mixed = ggml_scale(ctx, inp(ctx, b, n, 3, T), 1.5f);
            ggml_tensor * msgs  = ggml_scale(ctx, inp(ctx, b, n, 4, T), 0.7f);
            ggml_tensor * cur = ggml_cont(ctx, ggml_view_2d(ctx, mixed, n, T, mixed->nb[2], 1*mixed->nb[1]));
            ggml_tensor * msg = ggml_cont(ctx, ggml_view_2d(ctx, msgs,  n, T, msgs->nb[2],  2*msgs->nb[1]));
            ggml_tensor * down = inp(ctx, b, 2*n, r); ggml_tensor * wg = inp(ctx, b, r, n); ggml_tensor * wf = inp(ctx, b, r, n);
            ggml_tensor * dn = ggml_scale(ctx, down, 0.02f);
            ggml_tensor * out = control(ctx, cur, msg, dn, wg, wf);
            // mixed and msgs are evaluated first, as in the model, so the chain from the first CONT is unbroken
            b.outputs = { mixed, msgs, dn, ggml_scale(ctx, out, 1.0f), ggml_scale(ctx, wg, 1.0f), ggml_scale(ctx, wf, 1.0f), ggml_scale(ctx, mixed, 1.0f), ggml_scale(ctx, msgs, 1.0f) }; return b; }});
        v.push_back({"moew_T" + std::to_string(T), [T](ggml_context * ctx) {
            built b;
            ggml_tensor * w = ggml_sqr(ctx, inp(ctx, b, 8, T));
            ggml_tensor * s = ggml_scale_bias(ctx, ggml_sum_rows(ctx, w), 1.0f, 1e-20f);
            ggml_tensor * o = ggml_reshape_3d(ctx, ggml_div(ctx, w, s), 1, 8, T);
            b.outputs = { ggml_scale(ctx, o, 8.0f) }; return b; }});
    }
    for (int T : {1, 3}) for (int n_kv : {256, 1500}) for (int with_clip : {1, 0}) {
        v.push_back({std::string(with_clip ? "relbias_T" : "relbiasraw_T") + std::to_string(T) + "_kv" + std::to_string(n_kv), [T, n_kv, with_clip](ggml_context * ctx) {
            built b; const int64_t n_rel = 16, n_extent = 1024, n_head = 24;
            ggml_tensor * bias = inp(ctx, b, n_rel, n_extent);
            ggml_tensor * Rcur = ggml_scale(ctx, inp(ctx, b, n_rel, n_head, T), 3.0f);
            ggml_tensor * rs = with_clip ? inp(ctx, b, n_rel, n_head) : nullptr;
            ggml_tensor * sink_w = inp(ctx, b, 64, n_head); ggml_tensor * sink_x = ggml_scale(ctx, inp(ctx, b, 64, T), 1.0f);
            ggml_tensor * rel_pos = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv, T); ggml_set_input(rel_pos); b.inputs.push_back(rel_pos);
            ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, T); ggml_set_input(mask); b.inputs.push_back(mask);
            // as in build_layer + build_attn
            if (with_clip) Rcur = clip(ctx, Rcur, rs, 8.0f);
            ggml_tensor * sinks = ggml_mul_mat(ctx, sink_w, sink_x);
            ggml_tensor * table = ggml_mul_mat(ctx, ggml_pad(ctx, bias, 0, 1, 0, 0), Rcur);
            table = ggml_sub(ctx, table, ggml_reshape_3d(ctx, sinks, 1, n_head, T));
            table = ggml_cont(ctx, ggml_permute(ctx, table, 1, 0, 2, 3));
            ggml_tensor * kq_b = ggml_get_rows(ctx, table, rel_pos);
            kq_b = ggml_reshape_4d(ctx, kq_b, n_head, n_kv, T, 1);
            kq_b = ggml_cont(ctx, ggml_permute(ctx, kq_b, 2, 0, 1, 3));
            ggml_tensor * kq_mask = ggml_cast(ctx, mask, GGML_TYPE_F32);
            kq_mask = ggml_add(ctx, kq_b, kq_mask);
            kq_mask = ggml_cast(ctx, kq_mask, GGML_TYPE_F16);
            // the sink fill that build_layer places right behind; a nonzero value so a missing write shows
            ggml_tensor * fill = ggml_fill(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_head), 0.25f);
            // sink_x, rel_pos and mask are evaluated or shared before the chain in the model: keep them alive past it
            b.outputs = { sink_x, kq_mask, fill, ggml_scale(ctx, bias, 1.0f), ggml_cast(ctx, mask, GGML_TYPE_F32), ggml_cast(ctx, rel_pos, GGML_TYPE_F32),
                          ggml_scale(ctx, sink_x, 1.0f) };
            if (rs) b.outputs.push_back(ggml_scale(ctx, rs, 1.0f));
            return b; }});
    }
    for (int T : {1, 5}) for (int full : {2, 1, 0}) for (int C : {1024, 2304}) {
        v.push_back({std::string(full == 2 ? "convres" : full ? "convfull" : "convstore") + "_T" + std::to_string(T) + "_C" + std::to_string(C), [T, C, full](ggml_context * ctx) {
            built b; const int64_t n_conv = 4, Ctot = 6656, off = C == 1024 ? 1024 : 4352;
            // as convolve() + build_conv_state_store(): history is a gathered copy of the state, s is the state itself
            ggml_tensor * history = ggml_scale(ctx, inp(ctx, b, n_conv - 1, Ctot, 1), 1.0f);
            ggml_tensor * s = inp(ctx, b, (n_conv - 1)*Ctot, 1);
            ggml_tensor * w = inp(ctx, b, n_conv, C);
            ggml_tensor * cur = ggml_scale(ctx, inp(ctx, b, C, T), 2.0f);
            ggml_tensor * past = ggml_view_3d(ctx, history, n_conv - 1, C, 1, history->nb[1], history->nb[2], off*history->nb[1]);
            ggml_tensor * x = ggml_transpose(ctx, ggml_reshape_3d(ctx, cur, C, T, 1));
            x = ggml_concat(ctx, past, x, 0);
            ggml_tensor * last = ggml_view_3d(ctx, x, n_conv - 1, C, 1, x->nb[1], x->nb[2], ggml_row_size(x->type, T));
            ggml_tensor * cell = ggml_view_2d(ctx, s, (n_conv - 1)*C, 1, s->nb[1], off*(n_conv - 1)*sizeof(float));
            ggml_tensor * store = ggml_cpy(ctx, last, cell);
            ggml_tensor * out = ggml_reshape_2d(ctx, ggml_ssm_conv(ctx, x, w), C, T);
            out = ggml_scale(ctx, ggml_add(ctx, cur, out), 1.0f/sqrtf(n_conv + 1));
            // store first, as build_forward_expand does in the model; an unrelated node in between gives the K/V form
            if (full == 2) { // with the residual add of the layer
                ggml_tensor * res = ggml_scale(ctx, inp(ctx, b, C, T), 0.3f);
                b.outputs = { res, store, ggml_add(ctx, res, out), ggml_scale(ctx, w, 1.0f), ggml_scale(ctx, history, 1.0f), ggml_scale(ctx, res, 1.0f) };
            } else if (full) b.outputs = { store, out, ggml_scale(ctx, w, 1.0f), ggml_scale(ctx, history, 1.0f) };
            else      b.outputs = { store, ggml_scale(ctx, inp(ctx, b, 64), 3.0f), out, ggml_scale(ctx, w, 1.0f), ggml_scale(ctx, history, 1.0f) };
            return b; }});
    }
    for (int T : {1, 5}) {
        v.push_back({"rot_q_T" + std::to_string(T), [T](ggml_context * ctx) {
            built b; ggml_tensor * x = inp(ctx, b, 128, 24, T); ggml_tensor * s = inp(ctx, b, 128, 24); ggml_tensor * rot = inp(ctx, b, 128, 128);
            ggml_tensor * q = clip(ctx, ggml_scale(ctx, x, 9.0f), s, 22.627417f);
            b.outputs = { ggml_mul_mat(ctx, rot, q), ggml_scale(ctx, s, 1.0f) }; return b; }});
        v.push_back({"rot_plain_T" + std::to_string(T), [T](ggml_context * ctx) {
            built b; ggml_tensor * x = inp(ctx, b, 128, 24, T); ggml_tensor * rot = inp(ctx, b, 128, 128);
            b.outputs = { ggml_mul_mat(ctx, rot, ggml_scale(ctx, x, 2.0f)) }; return b; }});
        for (int rms : {0, 1}) {
            v.push_back({std::string(rms ? "rot_v_T" : "rot_k_T") + std::to_string(T), [T, rms](ggml_context * ctx) {
                built b; const int64_t n_conv = 4, C = 1024;
                ggml_tensor * x = ggml_scale(ctx, inp(ctx, b, n_conv - 1 + T, C, 1), 1.0f);
                ggml_tensor * w = inp(ctx, b, n_conv, C);
                ggml_tensor * cur = ggml_scale(ctx, inp(ctx, b, C, T), 20.0f);
                ggml_tensor * rot = inp(ctx, b, 128, 128);
                ggml_tensor * out = ggml_reshape_2d(ctx, ggml_ssm_conv(ctx, x, w), C, T);
                out = ggml_scale(ctx, ggml_add(ctx, cur, out), 1.0f/sqrtf(n_conv + 1));
                out = ggml_reshape_3d(ctx, out, 128, 8, T);
                out = rms ? ggml_rms_norm(ctx, out, 1e-6f) : clip(ctx, out, nullptr, 22.627417f);
                b.outputs = { ggml_mul_mat(ctx, rot, out), ggml_scale(ctx, w, 1.0f), ggml_scale(ctx, x, 1.0f) }; return b; }});
        }
    }
    v.push_back({"qcache", [](ggml_context * ctx) {
        built b; const int64_t n = 2304;
        auto qw = [&](int64_t rows) { ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, n, rows); ggml_set_input(t); b.inputs.push_back(t); return t; };
        ggml_tensor * x1 = ggml_scale(ctx, inp(ctx, b, n), 1.5f);
        ggml_tensor * x2 = ggml_scale(ctx, inp(ctx, b, n), 0.5f);
        // x1 feeds three matvecs with one of x2 in between, as Q, K, relative bias and gate do
        b.outputs = { ggml_mul_mat(ctx, qw(3072), x1), ggml_mul_mat(ctx, qw(1024), x2), ggml_mul_mat(ctx, qw(384), x1),
                      ggml_mul_mat(ctx, qw(24), x1), ggml_mul_mat(ctx, qw(64), x2) };
        return b; }});
    v.push_back({"qcache_one", [](ggml_context * ctx) {
        // one vector only: the cache entry of an evaluation matches the first matvec of the next one, with stale contents
        built b; const int64_t n = 2304;
        auto qw = [&](int64_t rows) { ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, n, rows); ggml_set_input(t); b.inputs.push_back(t); return t; };
        ggml_tensor * x1 = ggml_scale(ctx, inp(ctx, b, n), 1.5f);
        b.outputs = { ggml_mul_mat(ctx, qw(384), x1), ggml_mul_mat(ctx, qw(24), x1) };
        return b; }});
    for (int form : {0, 1}) {
        // single token, no copies: cur and msg are dense column views (attention), or cur is a plain tensor (FFN)
        v.push_back({std::string(form ? "controlview_ffn" : "controlview_attn"), [form](ggml_context * ctx) {
            built b; const int64_t n = 2304, r = 64;
            ggml_tensor * mixed = ggml_scale(ctx, inp(ctx, b, n, 3, 1), 1.5f);
            ggml_tensor * msgs  = ggml_scale(ctx, inp(ctx, b, n, 4, 1), 0.7f);
            ggml_tensor * cur = form ? ggml_reshape_2d(ctx, ggml_scale(ctx, inp(ctx, b, n, 1, 1), 1.2f), n, 1)
                                     : ggml_view_2d(ctx, mixed, n, 1, mixed->nb[1], 1*mixed->nb[1]);
            ggml_tensor * msg = ggml_view_2d(ctx, msgs, n, 1, msgs->nb[1], 2*msgs->nb[1]);
            ggml_tensor * down = inp(ctx, b, 2*n, r); ggml_tensor * wg = inp(ctx, b, r, n); ggml_tensor * wf = inp(ctx, b, r, n);
            ggml_tensor * dn = ggml_scale(ctx, down, 0.02f);
            ggml_tensor * out = control(ctx, cur, msg, dn, wg, wf);
            b.outputs = { mixed, msgs, dn };
            if (form) b.outputs.push_back(cur);
            for (ggml_tensor * t : { ggml_scale(ctx, out, 1.0f), ggml_scale(ctx, wg, 1.0f), ggml_scale(ctx, wf, 1.0f), ggml_scale(ctx, mixed, 1.0f), ggml_scale(ctx, msgs, 1.0f) }) b.outputs.push_back(t);
            return b; }});
    }
    for (int cpu_both : {0, 1}) for (int T : {1, 3}) for (int n_kv : {256, 1500}) {
        // the rel_bias_mask op (side under test) against the chain of ops it replaces (reference, on the CPU)
        test_case tc = {std::string(cpu_both ? "relopcpu_T" : "relop_T") + std::to_string(T) + "_kv" + std::to_string(n_kv), [T, n_kv](ggml_context * ctx) {
            built b; const int64_t n_rel = 16, n_extent = 1024, n_head = 24;
            ggml_tensor * bias = inp(ctx, b, n_rel, n_extent);
            ggml_tensor * Rcur = ggml_scale(ctx, inp(ctx, b, n_rel, n_head, T), 3.0f);
            ggml_tensor * rs = inp(ctx, b, n_rel, n_head);
            ggml_tensor * sink_w = inp(ctx, b, 64, n_head); ggml_tensor * sink_x = ggml_scale(ctx, inp(ctx, b, 64, T), 1.0f);
            ggml_tensor * rel_pos = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv, T); ggml_set_input(rel_pos); b.inputs.push_back(rel_pos);
            ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, T); ggml_set_input(mask); b.inputs.push_back(mask);
            Rcur = clip(ctx, Rcur, rs, 8.0f);
            ggml_tensor * sinks = ggml_mul_mat(ctx, sink_w, sink_x);
            ggml_tensor * kq_mask;
            if (g_test_side) {
                // as the model does: every layer writes one buffer; without a buffer for the single-token case
                kq_mask = ggml_rel_bias_mask(ctx, bias, Rcur, sinks, rel_pos, mask, T == 1 ? nullptr : ggml_new_tensor_3d(ctx, GGML_TYPE_F16, n_kv, T, n_head));
            } else {
                ggml_tensor * table = ggml_mul_mat(ctx, ggml_pad(ctx, bias, 0, 1, 0, 0), Rcur);
                table = ggml_sub(ctx, table, ggml_reshape_3d(ctx, sinks, 1, n_head, T));
                table = ggml_cont(ctx, ggml_permute(ctx, table, 1, 0, 2, 3));
                ggml_tensor * kq_b = ggml_get_rows(ctx, table, rel_pos);
                kq_b = ggml_reshape_4d(ctx, kq_b, n_head, n_kv, T, 1);
                kq_b = ggml_cont(ctx, ggml_permute(ctx, kq_b, 2, 0, 1, 3));
                kq_mask = ggml_add(ctx, kq_b, ggml_cast(ctx, mask, GGML_TYPE_F32));
                kq_mask = ggml_cast(ctx, kq_mask, GGML_TYPE_F16);
            }
            ggml_tensor * fill = ggml_fill(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_head), 0.25f);
            b.outputs = { sink_x, kq_mask, fill, ggml_scale(ctx, bias, 1.0f), ggml_cast(ctx, mask, GGML_TYPE_F32), ggml_cast(ctx, rel_pos, GGML_TYPE_F32),
                          ggml_scale(ctx, sink_x, 1.0f), ggml_scale(ctx, rs, 1.0f) };
            return b; }};
        tc.cpu_both = cpu_both;
        v.push_back(tc);
    }
    v.push_back({"mvrows", [](ggml_context * ctx) {
        // timing probe (kbench): quantized matvecs of several row counts on one vector
        built b; const int64_t n = 2304;
        auto qw = [&](int64_t rows) { ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, n, rows); ggml_set_input(t); b.inputs.push_back(t); return t; };
        ggml_tensor * x1 = ggml_scale(ctx, inp(ctx, b, n), 1.5f);
        for (int64_t rows : { 24, 384, 392, 1024, 2304, 3072 }) b.outputs.push_back(ggml_mul_mat(ctx, qw(rows), x1));
        return b; }});
    v.push_back({"mmvgroup", [](ggml_context * ctx) {
        // the projection stage of a looped layer: K, V, Q, relative bias, sinks and gate on three input vectors
        built b; const int64_t n = 2304;
        auto qw = [&](int64_t rows) { ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, n, rows); ggml_set_input(t); b.inputs.push_back(t); return t; };
        ggml_tensor * xq = ggml_scale(ctx, inp(ctx, b, n), 1.5f);
        ggml_tensor * xk = ggml_scale(ctx, inp(ctx, b, n), 0.5f);
        ggml_tensor * xv = ggml_scale(ctx, inp(ctx, b, n), 0.8f);
        b.outputs = { xq, xk, xv, ggml_mul_mat(ctx, qw(1024), xk), ggml_mul_mat(ctx, qw(1024), xv), ggml_mul_mat(ctx, qw(3072), xq),
                      ggml_mul_mat(ctx, qw(384), xq), ggml_mul_mat(ctx, qw(24), xq), ggml_mul_mat(ctx, qw(3072), xq), ggml_mul_mat(ctx, qw(31), xk) };
        return b; }});
    v.push_back({"normq", [](ggml_context * ctx) {
        // layer norm feeding quantized projections: the norm kernel hands them the q8_1 input
        built b; const int64_t n = 2304;
        auto qw = [&](int64_t rows) { ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, n, rows); ggml_set_input(t); b.inputs.push_back(t); return t; };
        ggml_tensor * x = ggml_scale(ctx, inp(ctx, b, n), 3.0f);
        ggml_tensor * w = inp(ctx, b, n);
        ggml_tensor * y = ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), w);
        b.outputs = { y, ggml_mul_mat(ctx, qw(384), y), ggml_mul_mat(ctx, qw(24), y), ggml_mul_mat(ctx, qw(1024), y), ggml_scale(ctx, w, 1.0f) };
        return b; }});
    for (int T : {1, 5}) for (int kv : {0, 1, 2}) {
        // kv 2: a full-attention layer, with input-dependent conv taps on K and V and the position factor on Q and r
        v.push_back({std::string(kv == 2 ? "postprojdyn_T" : kv ? "postprojkv_T" : "postproj_T") + std::to_string(T), [T, kv](ggml_context * ctx) {
            const bool dynamic = kv == 2;
            // the post-projection stage as build_layer orders it: r clip, Q clip, K conv+clip+store, V conv+rms+store, rotations
            built b; const int64_t n_conv = 4, C = 1024, Ctot = 6656;
            ggml_tensor * history = ggml_scale(ctx, inp(ctx, b, n_conv - 1, Ctot, 1), 1.0f);
            ggml_tensor * s = inp(ctx, b, (n_conv - 1)*Ctot, 1);
            ggml_tensor * Qraw = ggml_scale(ctx, inp(ctx, b, 3072, T), 9.0f);
            ggml_tensor * Kraw = ggml_scale(ctx, inp(ctx, b, C, T), 20.0f);
            ggml_tensor * Vraw = ggml_scale(ctx, inp(ctx, b, C, T), 2.0f);
            ggml_tensor * Rraw = ggml_scale(ctx, inp(ctx, b, 384, T), 4.0f);
            ggml_tensor * qs = inp(ctx, b, 128, 24); ggml_tensor * rs = inp(ctx, b, 16, 24);
            ggml_tensor * wk = inp(ctx, b, n_conv, C); ggml_tensor * wv = inp(ctx, b, n_conv, C);
            ggml_tensor * rotk = inp(ctx, b, 128, 128); ggml_tensor * rotv = inp(ctx, b, 128, 128);
            ggml_tensor * kdyn = dynamic ? ggml_scale(ctx, inp(ctx, b, n_conv*8, T), 0.5f) : nullptr;
            ggml_tensor * vdyn = dynamic ? ggml_scale(ctx, inp(ctx, b, n_conv*8, T), 0.5f) : nullptr;
            ggml_tensor * tau  = dynamic ? ggml_scale(ctx, inp(ctx, b, 1, 1, T), 1.3f) : nullptr;
            auto conv = [&](ggml_tensor * raw, ggml_tensor * w, ggml_tensor * dyn, int64_t off, ggml_tensor *& store) {
                ggml_tensor * past = ggml_view_3d(ctx, history, n_conv - 1, C, 1, history->nb[1], history->nb[2], off*history->nb[1]);
                ggml_tensor * x = ggml_concat(ctx, past, ggml_transpose(ctx, ggml_reshape_3d(ctx, raw, C, T, 1)), 0);
                ggml_tensor * last = ggml_view_3d(ctx, x, n_conv - 1, C, 1, x->nb[1], x->nb[2], ggml_row_size(x->type, T));
                ggml_tensor * cell = ggml_view_2d(ctx, s, (n_conv - 1)*C, 1, s->nb[1], off*(n_conv - 1)*sizeof(float));
                store = ggml_cpy(ctx, last, cell);
                ggml_tensor * out = ggml_reshape_2d(ctx, ggml_ssm_conv(ctx, x, w), C, T);
                if (dyn) { // as convolve(): one term per lag, the tap of a head times the window shifted by the lag
                    const int64_t n_heads = dyn->ne[0]/n_conv;
                    dyn = ggml_reshape_3d(ctx, dyn, n_conv, n_heads, T);
                    for (int64_t lag = 0; lag < n_conv; ++lag) {
                        ggml_tensor * tap    = ggml_view_3d(ctx, dyn, 1, n_heads, T, dyn->nb[1], dyn->nb[2], lag*dyn->nb[0]);
                        ggml_tensor * window = ggml_view_3d(ctx, x, T, C, 1, x->nb[1], x->nb[2], (n_conv - 1 - lag)*x->nb[0]);
                        window = ggml_cont(ctx, ggml_transpose(ctx, window));
                        window = ggml_reshape_3d(ctx, window, C/n_heads, n_heads, T);
                        out = ggml_add(ctx, out, ggml_reshape_2d(ctx, ggml_mul(ctx, window, tap), C, T));
                    }
                }
                return ggml_scale(ctx, ggml_add(ctx, raw, out), 1.0f/sqrtf((dyn ? 2 : 1)*n_conv + 1));
            };
            ggml_tensor * Kstore; ggml_tensor * Vstore;
            ggml_tensor * Kc = clip(ctx, ggml_reshape_3d(ctx, conv(Kraw, wk, kdyn, 0, Kstore), 128, 8, T), nullptr, 22.627417f);
            ggml_tensor * Vn = ggml_rms_norm(ctx, ggml_reshape_3d(ctx, conv(Vraw, wv, vdyn, C, Vstore), 128, 8, T), 1e-6f);
            ggml_tensor * Qc = clip(ctx, ggml_reshape_3d(ctx, Qraw, 128, 24, T), qs, 22.627417f);
            ggml_tensor * Rc = clip(ctx, ggml_reshape_3d(ctx, Rraw, 16, 24, T), rs, 8.0f);
            if (dynamic) {
                Qc = ggml_mul(ctx, Qc, tau);
                Rc = ggml_mul(ctx, Rc, tau);
            }
            ggml_tensor * rq = ggml_mul_mat(ctx, rotk, Qc); ggml_tensor * rk = ggml_mul_mat(ctx, rotk, Kc); ggml_tensor * rv = ggml_mul_mat(ctx, rotv, Vn);
            b.silent = { Qc, Kc, Vn };
            b.outputs = { history, Qraw, Kraw, Vraw, Rraw };
            if (dynamic) for (ggml_tensor * t : { kdyn, vdyn, tau }) b.outputs.push_back(t);
            for (ggml_tensor * t : { Rc, Qc, Kc, Kstore, Vn, Vstore, rq, rk, rv }) b.outputs.push_back(t);
            if (kv) {
                // the cache stores of build_attn: rotated K and V rows go into turbo3 caches at the token's row
                ggml_tensor * tok = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, T); ggml_set_input(tok); b.inputs.push_back(tok);
                for (ggml_tensor * rows : { rk, rv }) {
                    ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_TURBO3_0, C, 32); ggml_set_input(cache); b.inputs.push_back(cache);
                    b.outputs.push_back(ggml_set_rows(ctx, cache, ggml_view_2d(ctx, rows, C, T, rows->nb[2], 0), tok));
                }
                b.silent.push_back(rk); b.silent.push_back(rv);
            }
            for (ggml_tensor * t : { ggml_scale(ctx, qs, 1.0f), ggml_scale(ctx, rs, 1.0f), ggml_scale(ctx, wk, 1.0f), ggml_scale(ctx, wv, 1.0f) }) b.outputs.push_back(t);
            return b; }});
    }
    for (int T : {1, 5}) {
        v.push_back({"ctrlgroup_T" + std::to_string(T), [T](ggml_context * ctx) {
            // the input stage of a looped layer: three controls on columns of the mixed stream, then their three norms,
            // then projections that read the normed vectors (single token only for those)
            built b; const int64_t n = 2304, r = 64;
            ggml_tensor * mixed = ggml_scale(ctx, inp(ctx, b, n, 1, T), 1.5f);
            ggml_tensor * msgs  = ggml_scale(ctx, inp(ctx, b, n, 4, T), 0.7f);
            auto column = [&](ggml_tensor * t, int col) {
                if (T == 1) return ggml_view_2d(ctx, t, n, 1, t->nb[1], col*t->nb[1]);
                return ggml_cont(ctx, ggml_view_2d(ctx, t, n, T, t->nb[2], col*t->nb[1]));
            };
            b.outputs = { mixed, msgs };
            std::vector<ggml_tensor *> ctl, keep;
            for (int j = 0; j < 3; ++j) {
                ggml_tensor * dn = ggml_scale(ctx, inp(ctx, b, 2*n, r), 0.02f);
                ggml_tensor * wg = inp(ctx, b, r, n); ggml_tensor * wf = inp(ctx, b, r, n);
                b.outputs.push_back(dn);
                ctl.push_back(control(ctx, column(mixed, 0), column(msgs, j), dn, wg, wf));
                keep.push_back(ggml_scale(ctx, wg, 1.0f)); keep.push_back(ggml_scale(ctx, wf, 1.0f));
            }
            for (ggml_tensor * t : ctl) { b.outputs.push_back(t); b.silent.push_back(t); }
            std::vector<ggml_tensor *> normed;
            for (int j = 0; j < 3; ++j) {
                ggml_tensor * w = inp(ctx, b, n);
                normed.push_back(ggml_mul(ctx, ggml_rms_norm(ctx, ctl[j], 1e-6f), w));
                keep.push_back(ggml_scale(ctx, w, 1.0f));
            }
            for (ggml_tensor * t : normed) b.outputs.push_back(t);
            for (ggml_tensor * t : ctl) b.outputs.push_back(ggml_scale(ctx, t, 1.0f)); // as the model's keep-alive views
            if (T == 1) {
                auto qw = [&](int64_t rows) { ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, n, rows); ggml_set_input(t); b.inputs.push_back(t); return t; };
                for (int j = 0; j < 3; ++j) b.outputs.push_back(ggml_mul_mat(ctx, qw(384), normed[j]));
            }
            for (ggml_tensor * t : keep) b.outputs.push_back(t);
            for (ggml_tensor * t : { ggml_scale(ctx, mixed, 1.0f), ggml_scale(ctx, msgs, 1.0f) }) b.outputs.push_back(t);
            return b; }});
    }
    for (int T : {1, 5}) {
        v.push_back({"outgate_T" + std::to_string(T), [T](ggml_context * ctx) {
            // attention output: rotate back, multiply by the sigmoid gate, project (quantized weights for one token)
            built b; const int64_t n = 3072;
            ggml_tensor * att  = ggml_scale(ctx, inp(ctx, b, 128, 24, T), 2.0f);
            ggml_tensor * gate = ggml_scale(ctx, inp(ctx, b, n, T), 1.5f);
            ggml_tensor * rot  = inp(ctx, b, 128, 128);
            ggml_tensor * cur  = ggml_reshape_2d(ctx, ggml_mul_mat(ctx, rot, att), n, T);
            cur = ggml_mul(ctx, cur, ggml_sigmoid(ctx, gate));
            b.outputs = { att, gate, cur };
            if (T == 1) {
                ggml_tensor * wo = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, n, 2304); ggml_set_input(wo); b.inputs.push_back(wo);
                b.outputs.push_back(ggml_mul_mat(ctx, wo, cur));
            }
            b.outputs.push_back(ggml_scale(ctx, rot, 1.0f)); // a weight in the model: keep it alive past the chain
            return b; }});
    }
    for (int T : {1, 5}) {
        v.push_back({"route_T" + std::to_string(T), [T](ggml_context * ctx) {
            // build_moe_ffn's router with sigmoid gating and a selection bias, then the weight normalisation
            built b; const int64_t n_expert = 128, n_used = 8;
            ggml_tensor * logits = ggml_scale(ctx, inp(ctx, b, n_expert, T), 2.0f);
            ggml_tensor * bias = inp(ctx, b, n_expert);
            ggml_tensor * probs = ggml_sigmoid(ctx, logits);
            ggml_tensor * sel = ggml_add(ctx, probs, bias);
            ggml_tensor * ids = ggml_view_2d(ctx, ggml_argsort(ctx, sel, GGML_SORT_ORDER_DESC), n_used, T, n_expert*sizeof(int32_t), 0);
            ggml_tensor * w = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, n_expert, T), ids);
            w = ggml_reshape_2d(ctx, w, n_used, T);
            ggml_tensor * wsum = ggml_scale_bias(ctx, ggml_sum_rows(ctx, w), 1.0f, 1e-20f);
            w = ggml_reshape_3d(ctx, ggml_div(ctx, w, wsum), 1, n_used, T);
            w = ggml_scale(ctx, w, 2.5f);
            // the ids are read as floats so both backends compare; the bias is a weight in the model
            b.outputs = { logits, w, ggml_cast(ctx, ggml_cont(ctx, ids), GGML_TYPE_F32), ggml_scale(ctx, bias, 1.0f) };
            return b; }});
    }
    v.push_back({"ffntail", [](ggml_context * ctx) {
        // end of the MoE block for one token: weighted sum of the expert outputs, convolve(), residual add
        built b; const int64_t n_conv = 4, C = 2304, Ctot = 6656, off = 4352, n_used = 8;
        ggml_tensor * history = ggml_scale(ctx, inp(ctx, b, n_conv - 1, Ctot, 1), 1.0f);
        ggml_tensor * s = inp(ctx, b, (n_conv - 1)*Ctot, 1);
        ggml_tensor * w = inp(ctx, b, n_conv, C);
        ggml_tensor * experts = ggml_scale(ctx, inp(ctx, b, C, n_used, 1), 1.5f);
        ggml_tensor * weights = ggml_scale(ctx, inp(ctx, b, 1, n_used, 1), 0.3f);
        ggml_tensor * res = ggml_scale(ctx, inp(ctx, b, C, 1), 0.3f);
        ggml_tensor * weighted = ggml_mul(ctx, experts, weights);
        ggml_tensor * cur = nullptr;
        for (int64_t k = 0; k < n_used; ++k) { // as build_moe_ffn
            ggml_tensor * one = ggml_view_2d(ctx, weighted, C, 1, weighted->nb[2], k*weighted->nb[1]);
            cur = k == 0 ? one : ggml_add(ctx, cur, one);
        }
        ggml_tensor * past = ggml_view_3d(ctx, history, n_conv - 1, C, 1, history->nb[1], history->nb[2], off*history->nb[1]);
        ggml_tensor * x = ggml_concat(ctx, past, ggml_transpose(ctx, ggml_reshape_3d(ctx, cur, C, 1, 1)), 0);
        ggml_tensor * last = ggml_view_3d(ctx, x, n_conv - 1, C, 1, x->nb[1], x->nb[2], ggml_row_size(x->type, 1));
        ggml_tensor * cell = ggml_view_2d(ctx, s, (n_conv - 1)*C, 1, s->nb[1], off*(n_conv - 1)*sizeof(float));
        ggml_tensor * store = ggml_cpy(ctx, last, cell);
        ggml_tensor * out = ggml_reshape_2d(ctx, ggml_ssm_conv(ctx, x, w), C, 1);
        out = ggml_scale(ctx, ggml_add(ctx, cur, out), 1.0f/sqrtf(n_conv + 1));
        b.outputs = { history, experts, weights, res, store, ggml_add(ctx, res, out), ggml_scale(ctx, w, 1.0f), ggml_scale(ctx, history, 1.0f),
                      ggml_scale(ctx, res, 1.0f), ggml_scale(ctx, experts, 1.0f), ggml_scale(ctx, weights, 1.0f) };
        return b; }});
    v.push_back({"moeffn", [](ggml_context * ctx) {
        // the MoE block for one token, as build_moe_ffn: norm, router, gate/up projection of the chosen experts, SiTU,
        // down projection, weighted sum. The norm and the SiTU kernels hand the expert projections their q8_1 inputs.
        built b; const int64_t n = 2304, ff = 1152, n_exp = 16, k = 8;
        auto q3 = [&](int64_t n_in, int64_t n_out) { ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, n_in, n_out, n_exp); ggml_set_input(t); b.inputs.push_back(t); return t; };
        ggml_tensor * x = ggml_scale(ctx, inp(ctx, b, n), 3.0f);
        ggml_tensor * wn = inp(ctx, b, n);
        ggml_tensor * gate_inp = ggml_scale(ctx, inp(ctx, b, n, n_exp), 0.05f);
        ggml_tensor * bias = inp(ctx, b, n_exp);
        ggml_tensor * cur = ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), wn);
        ggml_tensor * probs = ggml_sigmoid(ctx, ggml_mul_mat(ctx, gate_inp, cur));
        ggml_tensor * ids = ggml_view_2d(ctx, ggml_argsort(ctx, ggml_add(ctx, probs, bias), GGML_SORT_ORDER_DESC), k, 1, n_exp*sizeof(int32_t), 0);
        ggml_tensor * w = ggml_reshape_2d(ctx, ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, n_exp, 1), ids), k, 1);
        w = ggml_reshape_3d(ctx, ggml_div(ctx, w, ggml_scale_bias(ctx, ggml_sum_rows(ctx, w), 1.0f, 1e-20f)), 1, k, 1);
        w = ggml_scale(ctx, w, 2.5f);
        ggml_tensor * gu = ggml_mul_mat_id(ctx, q3(n, 2*ff), ggml_reshape_3d(ctx, cur, n, 1, 1), ids);
        ggml_tensor * act = situ(ctx, gu, ff, 4.0f, 25.0f);
        ggml_tensor * down = ggml_mul_mat_id(ctx, q3(ff, n), act, ids);
        ggml_tensor * weighted = ggml_mul(ctx, down, w);
        ggml_tensor * out = nullptr;
        for (int64_t e = 0; e < k; ++e) {
            ggml_tensor * one = ggml_view_2d(ctx, weighted, n, 1, weighted->nb[2], e*weighted->nb[1]);
            out = e == 0 ? one : ggml_add(ctx, out, one);
        }
        // the router weights are placed before the expert projections, as the model's graph has them
        b.silent = { w };
        b.outputs = { x, gate_inp, w, ggml_scale(ctx, out, 1.0f), ggml_scale(ctx, wn, 1.0f), ggml_scale(ctx, bias, 1.0f) };
        return b; }});
    return v;
}
