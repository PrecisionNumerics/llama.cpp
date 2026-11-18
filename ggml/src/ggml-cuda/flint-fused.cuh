#pragma once

#include "common.cuh"

// Decode-time fusions for single-token MoE layers (Flint):
//
// ggml_cuda_flint_route: RMS_NORM -> MUL (norm weight) -> MUL_MAT (BF16 router, F32 activations) -> sigmoid -> + bias
//   -> top-k -> normalized weights, in one launch (the last block does the top-k). Also writes the normalized activations
//   (F32 and q8_1, registered with the q8_1 cache for the following expert matmul).

// ggml_cuda_flint_attn: Q/K RMS_NORM -> MUL (-> ROPE), SET_ROWS of K and V, FLASH_ATTN_EXT for one token, in one launch.
//   One block per Q head. The output is also written as q8_1 and registered with the q8_1 cache for the output projection.

#define FLINT_NEXPERT 256

#define FLINT_MAX_USED 8

// sigmoid + bias top-k of the FLINT_NEXPERT logits, bit-exact with topk_moe_cuda: the same selection (larger value first,
// ties to the lower index) and the same sum of the weights (lane e % WARP_SIZE adds its experts in selection order, then
// a warp reduction). All threads of the block call it; threads 0..FLINT_NEXPERT-1 do the work, with named barrier 1. Each
// expert is ranked in its group of WARP_SIZE, and the n_used best of each group are ranked against each other (every
// expert larger than one of the top n_used is among them). Slot k < n_used: ids_s[k], w_s[k] (shared memory). bias_v: the
// bias of expert tid (loaded early by the caller, see flint_topk_bias). flint_topk_select is the selection alone (the
// weights before the normalization), flint_topk_weight the normalized weight of a slot.
static __device__ __forceinline__ float flint_topk_bias(const float * bias) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    return tid < FLINT_NEXPERT ? __ldg(&bias[tid]) : 0.0f;
}

// slot k < n_used: ids_s[k] and sel_w[k], the weight before the normalization; ends with __syncthreads()
static __device__ __forceinline__ void flint_topk_select(const float * logits, const float bias_v, const int n_used,
        int * ids_s, float * sel_w) {
    constexpr int ng = FLINT_NEXPERT / WARP_SIZE;
    // key: the value (order of the float), then the lower index
    __shared__ unsigned long long keys[FLINT_NEXPERT];
    __shared__ unsigned long long cand[ng * FLINT_MAX_USED];
    __shared__ float              wts[FLINT_NEXPERT];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid >= FLINT_NEXPERT) {
        __syncthreads();
        return;
    }

    float wt = 1.f / (1.f + expf(-__ldcg(&logits[tid])));
    if (__isnanf(wt)) {
        wt = -FLT_MAX;
    }
    float sel = wt + bias_v;
    if (__isnanf(sel)) {
        sel = -FLT_MAX;
    }
    unsigned int kb = __float_as_uint(sel);
    kb = (kb << 1) == 0 ? 0x80000000u : (kb & 0x80000000u) ? ~kb : (kb | 0x80000000u); // -0 == +0
    const unsigned long long key = ((unsigned long long) kb << 32) | (unsigned int) (FLINT_NEXPERT - 1 - tid);
    keys[tid] = key;
    wts[tid]  = wt;
    __syncwarp();

    const unsigned long long * kg = keys + (tid / WARP_SIZE) * WARP_SIZE;
    int r = 0;
#pragma unroll
    for (int j = 0; j < WARP_SIZE; ++j) {
        r += kg[j] > key;
    }
    if (r < n_used) {
        cand[(tid / WARP_SIZE) * n_used + r] = key;
    }
    asm volatile("bar.sync 1, %0;" :: "n"(FLINT_NEXPERT) : "memory");

    // 4 threads per candidate
    const int nc = ng * n_used;
    const int c  = tid / 4;
    unsigned long long kc = 0;
    r = 0;
    if (c < nc) {
        kc = cand[c];
#pragma unroll
        for (int j = tid % 4; j < ng * FLINT_MAX_USED; j += 4) {
            r += j < nc && cand[j] > kc;
        }
    }
    r += __shfl_xor_sync(0xFFFFFFFF, r, 1);
    r += __shfl_xor_sync(0xFFFFFFFF, r, 2);
    if (c < nc && tid % 4 == 0 && r < n_used) {
        const int e = FLINT_NEXPERT - 1 - (int) (kc & 0xFFFFFFFFu);
        ids_s[r] = e;
        sel_w[r] = wts[e];
    }
    __syncthreads();
}

// normalized weight of slot lane < n_used of a selection (all lanes of one warp call it)
static __device__ __forceinline__ float flint_topk_weight(const int * ids_s, const float * sel_w, const int n_used,
        const bool with_norm, const float clamp_val, const float scale_val, const int lane) {
    float out_w = lane < n_used ? sel_w[lane] : 0.0f;
    if (with_norm) {
        float wt_sum = 0.0f;
#pragma unroll
        for (int k = 0; k < FLINT_MAX_USED; ++k) {
            if (k < n_used && ids_s[k] % WARP_SIZE == lane) {
                wt_sum += sel_w[k];
            }
        }
        wt_sum = warp_reduce_sum(wt_sum);
        wt_sum = max(wt_sum, clamp_val);
        out_w *= 1.0f / wt_sum;
    }
    return out_w * scale_val;
}

static __device__ __forceinline__ void flint_topk_block(const float * logits, const float bias_v, const int n_used,
        const bool with_norm, const float clamp_val, const float scale_val, int * ids_s, float * w_s) {
    __shared__ float sel_w[FLINT_MAX_USED];
    flint_topk_select(logits, bias_v, n_used, ids_s, sel_w);
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid < WARP_SIZE) {
        const float out_w = flint_topk_weight(ids_s, sel_w, n_used, with_norm, clamp_val, scale_val, tid);
        if (tid < n_used) {
            w_s[tid] = out_w;
        }
    }
    __syncthreads();
}

// launch the deferred top-k, if any (a consumer of the ids that does not compute it)
void ggml_cuda_flint_topk_flush(ggml_backend_cuda_context & ctx);

struct ggml_cuda_flint_route_args {
    const ggml_tensor * x;        // RMS_NORM input [n_embd, 1]
    const ggml_tensor * norm_w;   // MUL weight [n_embd]
    ggml_tensor *       xn;       // MUL output [n_embd, 1] (written as F32 and q8_1)
    const ggml_tensor * router;   // BF16 [n_embd, n_expert]
    const ggml_tensor * bias;     // F32 [n_expert]
    ggml_tensor *       ids;      // I32 [n_expert_used, 1], view of the argsort output
    ggml_tensor *       weights;  // F32 [n_expert_used]
    float               eps;
    float               clamp;    // lower bound of the weight sum (normalization)
    float               scale;
    bool                norm;     // normalize the selected weights
    bool                defer;    // leave the top-k to the next kernel (ctx.topk_deferred)
};

bool ggml_cuda_flint_route_supported(ggml_backend_cuda_context & ctx, const ggml_cuda_flint_route_args & args);
void ggml_cuda_flint_route(ggml_backend_cuda_context & ctx, const ggml_cuda_flint_route_args & args);

// ggml_cuda_flint_norm_mmvq: RMS_NORM -> MUL (norm weight) -> MUL_MAT (Q4_0) for one token: every block normalizes and
//   quantizes the activations itself (bit-exact with rms_norm_mul_q8_1_f32 + the mul_mat_vec_q small_k path).
bool ggml_cuda_flint_norm_mmvq_supported(const ggml_tensor * x, const ggml_tensor * norm_w, const ggml_tensor * mm);
void ggml_cuda_flint_norm_mmvq(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * norm_w,
        float eps, ggml_tensor * mm);

// ggml_cuda_flint_norm_glu: RMS_NORM -> MUL -> MUL_MAT (up), MUL_MAT (gate) -> SWIGLU for one token, as flint_norm_mmvq with
//   the gate fused like mul_mat_vec_q does it.
bool ggml_cuda_flint_norm_glu_supported(const ggml_tensor * x, const ggml_tensor * norm_w, const ggml_tensor * up,
        const ggml_tensor * gate, const ggml_tensor * glu);
void ggml_cuda_flint_norm_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * norm_w,
        float eps, const ggml_tensor * up, const ggml_tensor * gate, ggml_tensor * glu);

// ggml_cuda_flint_mmvq_add: MUL_MAT (Q4_0, activations from the q8_1 cache) -> ADD for one token, bit-exact with
//   mul_mat_vec_q (4 warps, 1 row per block) with the fused add; the weights are loaded before the dependency wait.
bool ggml_cuda_flint_mmvq_add_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, const ggml_tensor * add);
void ggml_cuda_flint_mmvq_add(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, ggml_tensor * add);

struct ggml_cuda_flint_attn_args {
    const ggml_tensor * q;        // F32 [D, n_head]    (Q RMS_NORM input)
    const ggml_tensor * k;        // F32 [D, n_head_kv] (K RMS_NORM input)
    const ggml_tensor * v;        // F32 [D, n_head_kv]
    const ggml_tensor * q_norm_w; // F32 [D]
    const ggml_tensor * k_norm_w; // F32 [D]
    float               eps_q;
    float               eps_k;
    const ggml_tensor * q_rope;   // ROPE nodes, nullptr if none
    const ggml_tensor * k_rope;
    const ggml_tensor * k_set;    // SET_ROWS into the K cache
    const ggml_tensor * v_set;    // SET_ROWS into the V cache
    ggml_tensor *       fa;       // FLASH_ATTN_EXT
    const ggml_tensor * fa_2d;    // [D*n_head, 1] reshape of the output, registered with the q8_1 cache (or nullptr)
};

bool ggml_cuda_flint_attn_supported(ggml_backend_cuda_context & ctx, const ggml_cuda_flint_attn_args & args);
void ggml_cuda_flint_attn(ggml_backend_cuda_context & ctx, const ggml_cuda_flint_attn_args & args);
