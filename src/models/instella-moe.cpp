#include "models.h"

// AMD Instella-MoE on the turboquant tree.
// Graph is DeepSeek-V2 MLA + MoE with two deltas vs deepseek2.cpp:
//   [TAG_INSTELLA_GATED_ATTN] sigmoid(attn_gate @ x) scales attention before wo
//   [TAG_INSTELLA_FARSKIP]    dual residual streams (res_full / res_nort)

llm_build_instella_moe::llm_build_instella_moe(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t n_embd_head_k = hparams.n_embd_head_k_mla();

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

    const uint32_t kv_lora_rank = hparams.n_lora_kv;

    // YaRN kq pre-scale — same as deepseek2 [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
    GGML_ASSERT(ext_factor >= 0.0f);
    const float attn_factor_org = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));
    const float mscale   = attn_factor_org * (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale = 1.0f * mscale * mscale / sqrtf(float(n_embd_head_k));

    ggml_tensor * cur;

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_k();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // [TAG_INSTELLA_FARSKIP] dual residual streams; layer 0 keeps them equal
    ggml_tensor * res_full = inpL;
    ggml_tensor * res_nort = inpL;

    for (int il = 0; il < n_layer; ++il) {
        const bool is_moe = (uint32_t) il >= hparams.n_layer_dense_lead;

        // [TAG_INSTELLA_FARSKIP] attention normalises the routed-free stream
        cur = build_norm(res_nort, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
            // [TAG_INSTELLA_GATED_ATTN] gate from post-attn_norm activations
            ggml_tensor * gate = build_lora_mm(model.layers[il].wqkv_gate, cur);
            cb(gate, "attn_gate_proj", il);

            ggml_tensor * q = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
            cb(q, "q", il);

            ggml_tensor * q_nope =
                ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                             ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
            cb(q_nope, "q_nope", il);

            ggml_tensor * q_pe = ggml_view_3d(
                ctx0, q, n_embd_head_qk_rope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                ggml_row_size(q->type, n_embd_head_k) * n_head, ggml_row_size(q->type, n_embd_head_qk_nope));
            cb(q_pe, "q_pe", il);

            ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
            cb(kv_cmpr_pe, "kv_cmpr_pe", il);

            ggml_tensor * kv_cmpr =
                ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                             ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
            cb(kv_cmpr, "kv_cmpr", il);

            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
            cb(k_pe, "k_pe", il);

            q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(q_pe, "q_pe", il);

            k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(k_pe, "k_pe", il);

            kv_cmpr = build_norm(kv_cmpr, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(kv_cmpr, "kv_cmpr", il);

            q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
            cb(q_nope, "q_nope_perm", il);

            ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, model.layers[il].wk_b, q_nope);
            cb(q_nope_absorbed, "q_nope_absorbed", il);

            q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
            cb(q_nope_absorbed, "q_nope_absorbed_perm", il);

            // rope first for in-place context shifting
            ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
            cb(Qcur, "Qcur", il);

            kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
            cb(kv_cmpr, "kv_cmpr_reshape", il);

            ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = kv_cmpr;
            cb(Vcur, "Vcur", il);

            // MLA absorption → MQA; wo applied after gate
            cur = build_attn(inp_attn,
                    NULL, NULL, NULL,
                    Qcur, Kcur, Vcur, nullptr, nullptr, model.layers[il].wv_b, kq_scale, il);
            cb(cur, "attn_out", il);

            if (il == n_layer - 1 && inp_out_ids) {
                cur      = ggml_get_rows(ctx0, cur,      inp_out_ids);
                gate     = ggml_get_rows(ctx0, gate,     inp_out_ids);
                res_full = ggml_get_rows(ctx0, res_full, inp_out_ids);
            }

            // [TAG_INSTELLA_GATED_ATTN]
            cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
            cb(cur, "attn_gated", il);

            cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
            cb(cur, "attn_o_proj", il);
        }

        ggml_tensor * attn_out = ggml_add(ctx0, cur, res_full);
        cb(attn_out, "attn_res", il);

        // [TAG_INSTELLA_FARSKIP] FFN normalises pre-attention residual (parallel to attn)
        cur = build_norm(res_full, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if (!is_moe) {
            cur = build_ffn(cur,
                model.layers[il].ffn_up, NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            res_full = ggml_add(ctx0, attn_out, cur);
            res_nort = res_full;
        } else {
            ggml_tensor * moe_out = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il,
                nullptr,
                model.layers[il].ffn_gate_up_exps);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * ffn_shexp =
                build_ffn(cur,
                    model.layers[il].ffn_up_shexp, NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            // [TAG_INSTELLA_FARSKIP] res_nort omits routed experts for next layer's attention
            res_nort = ggml_add(ctx0, attn_out, ffn_shexp);
            cb(res_nort, "l_out_no_routed", il);

            res_full = ggml_add(ctx0, res_nort, moe_out);
        }

        res_full = build_cvec(res_full, il);
        cb(res_full, "l_out", il);
    }

    cur = build_norm(res_full, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}


void llama_model_instella_moe::load_arch_tensors(llama_model_loader & ml) {
    llama_model_deepseek2::load_arch_tensors(ml);

    // [TAG_INSTELLA_GATED_ATTN] learned gate before o_proj (fork)
    const int64_t n_embd            = hparams.n_embd;
    const int64_t n_head            = hparams.n_head();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();
    for (int i = 0; i < (int) hparams.n_layer; ++i) {
        layers[i].wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_head * n_embd_head_v_mla}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_instella_moe::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<llm_build_instella_moe>(*this, params);
}
