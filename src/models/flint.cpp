#include "models.h"

void llama_model_flint::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func);

    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp);
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);
    ml.get_arr(LLM_KV_ATTENTION_ROPE_PATTERN, hparams.rope_pattern);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    type = LLM_TYPE_UNKNOWN;
}

void llama_model_flint::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_ff_exp        = hparams.n_ff_exp();

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    expert_groups.assign(n_layer, {});
    shared_experts.assign(n_layer, {});

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.attn_sinks = create_tensor(tn(LLM_TENSOR_ATTN_SINKS, "weight", i), {n_head}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
        } else {
            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);

            if (n_expert == 0) {
                throw std::runtime_error("n_expert must be > 0");
            }
            if (n_expert_used == 0) {
                throw std::runtime_error("n_expert_used must be > 0");
            }

            // MoE branch: either one tensor per projection, or mixed-precision expert groups
            int64_t n_grouped = 0;
            for (int g = 0; g < GGML_MUL_MAT_ID_MULTI_MAX; ++g) {
                const std::string w = "g" + std::to_string(g) + ".weight";
                const ggml_tensor * meta = ml.get_tensor_meta(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, w.c_str(), i).str().c_str());
                if (meta == nullptr) {
                    break;
                }
                const int64_t n_group_exp = meta->ne[2];
                expert_group grp;
                grp.gate_up = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, w.c_str(), i), {  n_embd, 2*n_ff_exp, n_group_exp}, 0);
                grp.down    = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,    w.c_str(), i), {n_ff_exp,     n_embd, n_group_exp}, 0);
                expert_groups[i].push_back(grp);
                n_grouped += n_group_exp;
            }
            if (n_grouped > n_expert) {
                throw std::runtime_error("expert groups hold more than n_expert experts");
            }
            if (!expert_groups[i].empty() && n_grouped < n_expert && layer.ffn_exp_probs_b == nullptr) {
                throw std::runtime_error("pruned experts need ffn_exp_probs_b to keep them from being selected");
            }
            if (expert_groups[i].empty()) {
                layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
                layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, 0);
                layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
            }

            // Shared expert branch: with expert groups it can be split into experts of n_ff_exp, run by the same op
            const ggml_tensor * sh_meta = ml.get_tensor_meta(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "shared.weight", i).str().c_str());
            if (sh_meta != nullptr && !expert_groups[i].empty()) {
                const int64_t n_sh = sh_meta->ne[2];
                if (n_sh * n_ff_exp != (int64_t) hparams.n_ff_shexp) {
                    throw std::runtime_error("shared experts do not add up to the shared expert");
                }
                shared_experts[i].gate_up = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "shared.weight", i), {  n_embd, 2*n_ff_exp, n_sh}, 0);
                shared_experts[i].down    = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,    "shared.weight", i), {n_ff_exp,     n_embd, n_sh}, 0);
            } else {
                layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
                layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        hparams.n_ff_shexp, n_embd}, 0);
                layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
            }
        }
    }
}

llama_rope_params llama_model_flint::get_rope_params(const llama_cparams & cparams, int il) const {
    auto res = llama_model::get_rope_params(cparams, il);

    // the SWA layers use the global base and scale
    res.freq_base  = cparams.rope_freq_base;
    res.freq_scale = cparams.rope_freq_scale;

    return res;
}

std::unique_ptr<llm_graph_context> llama_model_flint::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

llama_model_flint::graph::graph(const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv_iswa();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
            Qcur = build_rope(Qcur, inp_pos, il);
            Kcur = build_rope(Kcur, inp_pos, il);
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, model.layers[il].attn_sinks, nullptr, 1.0f / sqrtf(float(n_embd_head)), il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // MoE branch
        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                    model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
                    model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
                    model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            const auto & flint  = static_cast<const llama_model_flint &>(model);
            const auto & groups = flint.expert_groups[il];
            const auto & shared = flint.shared_experts[il];
            ggml_tensor * moe_out = !groups.empty() ? build_grouped_moe(cur, groups, shared, il) : build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il, nullptr, nullptr,
                model.layers[il].ffn_up_exps_s,
                model.layers[il].ffn_gate_exps_s,
                model.layers[il].ffn_down_exps_s);
            cb(moe_out, "ffn_moe_out", il);

            if (shared.gate_up) {
                cur = moe_out; // the shared expert is in moe_out
            } else {
                ggml_tensor * ffn_shexp =
                    build_ffn(cur,
                        model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                        model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                        model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(ffn_shexp, "ffn_shexp", il);

                // grouped experts: shared expert first in the graph, so cur stays alive and the fused gate/up/glu matvec output does not alias it
                cur = groups.empty() ? ggml_add(ctx0, moe_out, ffn_shexp) : ggml_add(ctx0, ffn_shexp, moe_out);
            }
            cb(cur, "ffn_out", il);
        }
        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

// The router (and its fused top-k) over all experts, then one mul_mat_id_multi over the groups for
// gate_up with the swiglu, and one for down with the weighted sum over the selected experts. The shared
// experts, if given, run in the same ops.
ggml_tensor * llama_model_flint::graph::build_grouped_moe(ggml_tensor * cur, const std::vector<expert_group> & groups,
        const expert_group & shared, int il) {
    const auto & layer = model.layers[il];
    const int64_t n_tokens = cur->ne[1];

    auto [selected, weights] = build_moe_routing(cur, layer.ffn_gate_inp, nullptr, layer.ffn_exp_probs_b,
        n_expert, n_expert_used, hparams.expert_weights_norm, hparams.expert_weights_scale,
        (llama_expert_gating_func_type) hparams.expert_gating_func, il);

    ggml_tensor * gate_up[GGML_MUL_MAT_ID_MULTI_MAX];
    ggml_tensor * down[GGML_MUL_MAT_ID_MULTI_MAX];
    const int n_groups = (int) groups.size();
    for (int g = 0; g < n_groups; ++g) {
        gate_up[g] = groups[g].gate_up;
        down[g]    = groups[g].down;
    }

    ggml_tensor * x = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, n_tokens);

    ggml_tensor * act = ggml_mul_mat_id_multi(ctx0, gate_up, n_groups, x, selected, nullptr, shared.gate_up, true); // [n_ff_exp, n_slots, n_tokens]
    cb(act, "ffn_moe_swiglu", il);

    weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
    // [n_embd, n_tokens, 1]: no reshape, so the residual add follows the op directly and the CUDA backend can fuse it
    return ggml_mul_mat_id_multi(ctx0, down, n_groups, act, selected, weights, shared.down, false);
}
