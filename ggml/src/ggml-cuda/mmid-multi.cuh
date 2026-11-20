#pragma once

#include "common.cuh"
#include "vecdotq.cuh"

// GGML_OP_MUL_MAT_ID_MULTI: mul_mat_id over experts split into tensors of different types
bool ggml_cuda_mul_mat_id_multi_supported(const ggml_tensor * op);

// res/add_dst: fuse a following residual add (add_dst = dst + res) into a weighted op; dst itself is then not written
void ggml_cuda_mul_mat_id_multi(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * res = nullptr,
        ggml_tensor * add_dst = nullptr);

// tokens from which the experts are grouped (their weights are read once for all tokens); fewer tokens: per-token kernels,
// which also take a deferred router top-k (ggml_cuda_topk_deferred). With 8 tokens the 48 pairs of 6 routed slots hit ~44
// experts, too few shared ones for the grouped kernels to win.
#define MMID_GROUPED_MIN_TOKENS 9

// whether the weighted op `node` can write add_dst = node + res directly (see ggml_cuda_mul_mat_id_multi)
bool ggml_cuda_mul_mat_id_multi_can_fuse_add(const ggml_tensor * node, const ggml_tensor * res, const ggml_tensor * add_dst);

// device helpers shared with the fused single-token MoE kernel (flint-fused.cu)

typedef float (*mmid_multi_vec_dot_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

struct mmid_multi_experts {
    const char * data[GGML_MUL_MAT_ID_MULTI_MAX + 1];
    int64_t      nb1[GGML_MUL_MAT_ID_MULTI_MAX + 1];      // bytes per row
    int64_t      nb2[GGML_MUL_MAT_ID_MULTI_MAX + 1];      // bytes per expert
    int          type[GGML_MUL_MAT_ID_MULTI_MAX + 1];
    int          first[GGML_MUL_MAT_ID_MULTI_MAX + 1];    // first global expert id of each routed tensor
    int          n_as;                                    // routed tensors; entry n_as is the shared tensor
};

// vec_dot_q4_0_q8_1 subtracts the q4_0 offset with the block sum that q8_1 stores as F16, which overflows for
// activations above ~2000 (32 values of a block) and gives inf/NaN; this sums the int8 values instead (exact)
static __device__ __forceinline__ float mmid_multi_vec_dot_q4_0_q8_1(
        const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_q4_0 * bq4_0 = (const block_q4_0 *) vbq + kbx;

    int sumi = 0;
    int sumu = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        const int v  = get_int_b2(bq4_0->qs, iqs + i);
        const int u0 = get_int_b4(bq8_1->qs, iqs + i);
        const int u1 = get_int_b4(bq8_1->qs, iqs + i + QI4_0);
        sumi = ggml_cuda_dp4a((v >> 0) & 0x0F0F0F0F, u0, sumi);
        sumi = ggml_cuda_dp4a((v >> 4) & 0x0F0F0F0F, u1, sumi);
        sumu = ggml_cuda_dp4a(0x01010101, u0, sumu);
        sumu = ggml_cuda_dp4a(0x01010101, u1, sumu);
    }
    return __half2float(bq4_0->d) * __low2float(bq8_1->ds) * (sumi - 8*sumu);
}

// partial dot product of one row of x (type) and one q8_1 row of y: items item0, item0 + istep, ...
template <ggml_type type>
static __device__ __forceinline__ float mmid_multi_row_dot(const char * x_row, const block_q8_1 * y, const int ncols,
        const int item0, const int istep) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int qi = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr =
        type == GGML_TYPE_Q1_0 ? VDR_Q1_0_Q8_1_MMVQ :
        type == GGML_TYPE_Q2_0 ? VDR_Q2_0_Q8_1_MMVQ :
        type == GGML_TYPE_Q4_0 ? VDR_Q4_0_Q8_1_MMVQ :
                                 VDR_Q8_0_Q8_1_MMVQ;
    constexpr mmid_multi_vec_dot_t vec_dot =
        type == GGML_TYPE_Q1_0 ? vec_dot_q1_0_q8_1 :
        type == GGML_TYPE_Q2_0 ? vec_dot_q2_0_q8_1 :
        type == GGML_TYPE_Q4_0 ? mmid_multi_vec_dot_q4_0_q8_1 :
                                 vec_dot_q8_0_q8_1;
    constexpr int items_per_block = qi / vdr;

    const int n_items = (ncols / qk) * items_per_block;
    float sum = 0.0f;
    for (int item = item0; item < n_items; item += istep) {
        const int kbx = item / items_per_block;
        const int kqs = vdr * (item % items_per_block);
        sum += vec_dot(x_row, &y[kbx * (qk / QK8_1)], kbx, kqs);
    }
    return sum;
}

template <ggml_type type>
static __device__ __forceinline__ void mmid_multi_rows_dot(
        const char * xe, const int64_t nb1, const block_q8_1 * y, const int ncols, const int item0, const int istep,
        const int row, const int row_up, float & dot, float & dot_up, const bool swiglu) {
    if (!swiglu) {
        dot = mmid_multi_row_dot<type>(xe + row * nb1, y, ncols, item0, istep);
        return;
    }
    // both rows in one loop: their loads are in flight together (each sum in the order of mmid_multi_row_dot)
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int qi = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr =
        type == GGML_TYPE_Q1_0 ? VDR_Q1_0_Q8_1_MMVQ :
        type == GGML_TYPE_Q2_0 ? VDR_Q2_0_Q8_1_MMVQ :
        type == GGML_TYPE_Q4_0 ? VDR_Q4_0_Q8_1_MMVQ :
                                 VDR_Q8_0_Q8_1_MMVQ;
    constexpr mmid_multi_vec_dot_t vec_dot =
        type == GGML_TYPE_Q1_0 ? vec_dot_q1_0_q8_1 :
        type == GGML_TYPE_Q2_0 ? vec_dot_q2_0_q8_1 :
        type == GGML_TYPE_Q4_0 ? mmid_multi_vec_dot_q4_0_q8_1 :
                                 vec_dot_q8_0_q8_1;
    constexpr int items_per_block = qi / vdr;

    const int n_items = (ncols / qk) * items_per_block;
    const char * x_gate = xe + row * nb1;
    const char * x_up   = xe + row_up * nb1;
    dot    = 0.0f;
    dot_up = 0.0f;
#pragma unroll 1
    for (int item = item0; item < n_items; item += istep) {
        const int kbx = item / items_per_block;
        const int kqs = vdr * (item % items_per_block);
        dot    += vec_dot(x_gate, &y[kbx * (qk / QK8_1)], kbx, kqs);
        dot_up += vec_dot(x_up,   &y[kbx * (qk / QK8_1)], kbx, kqs);
    }
}

// partial dot products of rows row .. row + rpw - 1 of x (type) and one q8_1 row of y, loads of all rows in flight together;
// swiglu: also of the up rows row_up .. row_up + rpw - 1
template <ggml_type type, int rpw>
static __device__ __forceinline__ void mmid_multi_rows_dot_n(const char * xe, const int64_t nb1, const block_q8_1 * y,
        const int ncols, const int item0, const int istep, const int row, const int row_up, float * dot, float * dot_up,
        const bool swiglu) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int qi = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr =
        type == GGML_TYPE_Q1_0 ? VDR_Q1_0_Q8_1_MMVQ :
        type == GGML_TYPE_Q2_0 ? VDR_Q2_0_Q8_1_MMVQ :
        type == GGML_TYPE_Q4_0 ? VDR_Q4_0_Q8_1_MMVQ :
                                 VDR_Q8_0_Q8_1_MMVQ;
    constexpr mmid_multi_vec_dot_t vec_dot =
        type == GGML_TYPE_Q1_0 ? vec_dot_q1_0_q8_1 :
        type == GGML_TYPE_Q2_0 ? vec_dot_q2_0_q8_1 :
        type == GGML_TYPE_Q4_0 ? mmid_multi_vec_dot_q4_0_q8_1 :
                                 vec_dot_q8_0_q8_1;
    constexpr int items_per_block = qi / vdr;

    const int n_items = (ncols / qk) * items_per_block;
#pragma unroll
    for (int r = 0; r < rpw; ++r) {
        dot[r]    = 0.0f;
        dot_up[r] = 0.0f;
    }
    for (int item = item0; item < n_items; item += istep) {
        const int kbx = item / items_per_block;
        const int kqs = vdr * (item % items_per_block);
#pragma unroll
        for (int r = 0; r < rpw; ++r) {
            dot[r] += vec_dot(xe + (row + r) * nb1, &y[kbx * (qk / QK8_1)], kbx, kqs);
            if (swiglu) {
                dot_up[r] += vec_dot(xe + (row_up + r) * nb1, &y[kbx * (qk / QK8_1)], kbx, kqs);
            }
        }
    }
}

// expert data of slot s of a token (ids: its expert ids) and its tensor index g, or nullptr for a skipped slot
static __device__ __forceinline__ const char * mmid_multi_slot_expert(const mmid_multi_experts & experts,
        const int32_t * ids, const int s, const int n_used, int & g) {
    int e;
    if (s < n_used) {
        const int id = ids[s];
        if (id < 0 || id >= experts.first[experts.n_as]) {
            return nullptr;
        }
        g = 0;
        while (g + 1 < experts.n_as && id >= experts.first[g + 1]) {
            ++g;
        }
        e = id - experts.first[g];
    } else {
        g = experts.n_as; // shared expert
        e = s - n_used;
    }
    return experts.data[g] + (int64_t) e * experts.nb2[g];
}
