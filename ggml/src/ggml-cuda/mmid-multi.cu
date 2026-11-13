#include "mmid-multi.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

// GGML_OP_MUL_MAT_ID_MULTI
//
// Few tokens: one kernel for all expert tensors. For each slot a warp looks up the tensor of the slot's expert and
// runs that type's q8_1 dot product, so experts of different types share a launch. Several warps share one output
// row: over the slots for the weighted sum, over the columns otherwise. The swiglu and the weighted sum over the
// slots are applied in the same kernel.
//
// More tokens: one MMQ mul_mat_id per tensor, each on the slots of its experts (the others are skipped), all
// writing into one buffer, then a kernel for the swiglu or the weighted sum.
//
// Shared experts (optional) take the slots after the routed ones, with weight 1.

#define MMID_MULTI_MAX_BATCH 32 // tokens up to which the single-kernel path is used (RTX 5090: MMQ wins from ~48)
#define MMID_MULTI_NWARPS    4 // warps per block of the single-kernel path

static constexpr bool mmid_multi_type_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

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
    dot = mmid_multi_row_dot<type>(xe + row * nb1, y, ncols, item0, istep);
    if (swiglu) {
        dot_up = mmid_multi_row_dot<type>(xe + row_up * nb1, y, ncols, item0, istep);
    }
}

// expert data of slot s of token t and its tensor index g, or nullptr for a skipped slot
static __device__ __forceinline__ const char * mmid_multi_slot_expert(const mmid_multi_experts & experts,
        const int32_t * ids, const int s, const int t, const int n_used, const int64_t ids_s1, int & g) {
    int e;
    if (s < n_used) {
        const int id = ids[s + t * ids_s1];
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

// grid: (row blocks, sum_slots ? n_tokens : n_slots * n_tokens); block: (WARP_SIZE, MMID_MULTI_NWARPS)
// ksplit warps share one output row: they take every ksplit-th slot (sum_slots) or every ksplit-th 32 items
//
// PDL: the next kernel may launch at once; everything (ids, y, w) is read after the dependency wait.
template <bool swiglu, bool sum_slots, int ksplit>
static __global__ void __launch_bounds__(WARP_SIZE*MMID_MULTI_NWARPS) mul_mat_id_multi_vec(
        const mmid_multi_experts experts, const block_q8_1 * y, const int32_t * ids,
        const float * w, float * dst,
        const int ncols, const int rows_out, const int n_used, const int n_slots, const int64_t y_row_blocks,
        const bool y_bcast, const int64_t ids_s1, const int64_t dst_s1, const int64_t dst_s2,
        const float * res, const int64_t res_s1) {
    static_assert(MMID_MULTI_NWARPS % ksplit == 0, "ksplit must divide the warps of a block");
    constexpr int rows_per_block = MMID_MULTI_NWARPS / ksplit;

    const int lane = threadIdx.x;
    const int ks   = threadIdx.y % ksplit;
    const int row  = blockIdx.x * rows_per_block + threadIdx.y / ksplit;

    const int t       = sum_slots ? blockIdx.y : blockIdx.y / n_slots;
    const int s_begin = sum_slots ? ks : blockIdx.y % n_slots;
    const int s_end   = sum_slots ? n_slots : s_begin + 1;
    const int s_step  = sum_slots ? ksplit : 1;
    const int item0   = sum_slots ? lane : lane + ks * WARP_SIZE;
    const int istep   = sum_slots ? WARP_SIZE : WARP_SIZE * ksplit;

    ggml_cuda_pdl_lc();

    // the shared experts do not depend on the routing: prefetch this block's rows of them before the wait
    if (n_slots > n_used) {
        const int g      = experts.n_as;
        const int r0     = blockIdx.x * rows_per_block;
        const int nr     = min(rows_per_block, rows_out - r0);
        const int tid    = threadIdx.y * WARP_SIZE + threadIdx.x;
        const int nth    = WARP_SIZE * MMID_MULTI_NWARPS;
        const int s_lo   = sum_slots ? n_used : blockIdx.y % n_slots;
        const int s_hi   = sum_slots ? n_slots : s_lo + 1;
        for (int s = max(s_lo, n_used); s < s_hi && nr > 0; ++s) {
            const char * xe = experts.data[g] + (int64_t) (s - n_used) * experts.nb2[g];
            ggml_cuda_prefetch_l2(xe + (int64_t) r0 * experts.nb1[g], (int64_t) nr * experts.nb1[g], tid, nth);
            if (swiglu) {
                ggml_cuda_prefetch_l2(xe + (int64_t) (r0 + rows_out) * experts.nb1[g], (int64_t) nr * experts.nb1[g], tid, nth);
            }
        }
    }

    ggml_cuda_pdl_sync();

    float acc    = 0.0f;
    float acc_up = 0.0f;
    for (int s = s_begin; row < rows_out && s < s_end; s += s_step) {
        int g;
        const char * xe = mmid_multi_slot_expert(experts, ids, s, t, n_used, ids_s1, g);
        if (!xe) {
            continue; // skipped slot
        }
        const block_q8_1 * ys  = y + (y_bcast ? t : s + (int64_t) t * n_slots) * y_row_blocks;
        const int64_t      nb1 = experts.nb1[g];

        float dot    = 0.0f;
        float dot_up = 0.0f;
        switch (experts.type[g]) {
            case GGML_TYPE_Q1_0: mmid_multi_rows_dot<GGML_TYPE_Q1_0>(xe, nb1, ys, ncols, item0, istep, row, row + rows_out, dot, dot_up, swiglu); break;
            case GGML_TYPE_Q2_0: mmid_multi_rows_dot<GGML_TYPE_Q2_0>(xe, nb1, ys, ncols, item0, istep, row, row + rows_out, dot, dot_up, swiglu); break;
            case GGML_TYPE_Q4_0: mmid_multi_rows_dot<GGML_TYPE_Q4_0>(xe, nb1, ys, ncols, item0, istep, row, row + rows_out, dot, dot_up, swiglu); break;
            default:             mmid_multi_rows_dot<GGML_TYPE_Q8_0>(xe, nb1, ys, ncols, item0, istep, row, row + rows_out, dot, dot_up, swiglu); break;
        }
        // the slot sum is linear, so the lanes' partial sums are weighted before the reduction
        acc    += sum_slots && s < n_used ? w[s + t * n_used] * dot : dot;
        acc_up += dot_up;
    }

    acc = warp_reduce_sum(acc);
    if (swiglu) {
        acc_up = warp_reduce_sum(acc_up);
    }
    bool writer = lane == 0 && row < rows_out;
    if constexpr (ksplit > 1) {
        __shared__ float part[MMID_MULTI_NWARPS];
        __shared__ float part_up[MMID_MULTI_NWARPS];
        if (lane == 0) {
            part[threadIdx.y]    = acc;
            part_up[threadIdx.y] = acc_up;
        }
        __syncthreads();
        writer = writer && ks == 0;
        if (ks == 0) {
#pragma unroll
            for (int k = 1; k < ksplit; ++k) {
                acc    += part[threadIdx.y + k];
                acc_up += part_up[threadIdx.y + k];
            }
        }
    }
    if (writer) {
        const float out = swiglu ? ggml_cuda_op_silu_single(acc) * acc_up : acc;
        if (sum_slots) {
            // res: a fused residual add (may alias dst: this thread reads it before the write)
            dst[row + t * dst_s1] = res ? out + res[row + t * res_s1] : out;
        } else {
            dst[row + s_begin * dst_s1 + t * dst_s2] = out;
        }
    }

}

// local ids [n_slots, n_tokens] of one tensor. Routed: global id in [first, last) -> id - first, else -1.
// Shared: slot n_used + j -> j, routed slots -1.
static __global__ void mmid_multi_local_ids(const int32_t * __restrict__ ids, int32_t * __restrict__ ids_local,
        const int n_used, const int n_slots, const int64_t ids_s1, const int n, const int first, const int last,
        const bool shared) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) n_slots * n) {
        return;
    }
    const int s = i % n_slots;
    if (shared || s >= n_used) {
        ids_local[i] = shared && s >= n_used ? s - n_used : -1;
        return;
    }
    const int id = ids[s + (i / n_slots) * ids_s1];
    ids_local[i] = id >= first && id < last ? id - first : -1;
}

// tmp [rows, n_slots, n_tokens] -> silu(rows [0, half)) * rows [half, rows)
static __global__ void mmid_multi_swiglu(const float * __restrict__ tmp, float * __restrict__ dst, const int half,
        const int64_t n_cols, const int64_t dst_s1) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half * n_cols) {
        return;
    }
    const int64_t r = i % half;
    const int64_t c = i / half;
    dst[r + c * dst_s1] = ggml_cuda_op_silu_single(tmp[r + c * 2 * half]) * tmp[r + half + c * 2 * half];
}

// tmp [rows, n_slots, n_tokens] -> sum over slots of w * tmp (shared slots: weight 1) [+ res] -> [rows, n_tokens]
// res is a fused residual add; it may alias dst (each element is read before it is written, by the same thread)
static __global__ void mmid_multi_weighted_sum(const float * __restrict__ tmp, const float * __restrict__ w,
        float * dst, const int rows, const int n_used, const int n_slots, const int64_t n_tokens,
        const int64_t dst_s1, const float * res, const int64_t res_s1) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n_tokens) {
        return;
    }
    const int64_t r = i % rows;
    const int64_t t = i / rows;
    float sum = 0.0f;
    for (int s = 0; s < n_slots; ++s) {
        const float ws = s < n_used ? w[s + t * n_used] : 1.0f;
        sum += ws * tmp[r + (s + t * n_slots) * rows];
    }
    if (res) {
        sum += res[r + t * res_s1];
    }
    dst[r + t * dst_s1] = sum;
}

// ---------------------------------------------------------------------------------------------------------------------
// Grouped path (several tokens): expert-centric. The (token, slot) pairs are sorted by expert and split into segments
// of at most MMID_GROUPED_SEG pairs. A block takes one segment and a tile of output rows: each 32-value chunk of a
// weight row is unpacked to int8 once and multiplied (dp4a) with the q8_1 activations of up to MMID_GROUPED_NT
// tokens, so every selected expert is read once per batch instead of once per token. The type is chosen per block,
// so all precision groups and the shared experts run in one launch.

#define MMID_GROUPED_NT     8    // tokens per pass over a weight chunk (accumulators per thread)
#define MMID_GROUPED_SEG    32   // max pairs per segment (load balance: shared experts take every token)
#define MMID_GROUPED_NWARPS 4
#define MMID_GROUPED_MAX_E  1024 // max experts (routed + shared) in the sort kernel

// 32 values (chunk c of a row) as 8 ints of signed int8 and their scale
template <ggml_type type>
static __device__ __forceinline__ void mmid_grouped_load_chunk(const char * __restrict__ row, const int c, int v[8], float & d) {
    if constexpr (type == GGML_TYPE_Q1_0) {
        const block_q1_0 * b = (const block_q1_0 *) row + c / 4;
        d = __half2float(b->d);
        const int16_t * qs = (const int16_t *) b->qs + (c % 4) * 2;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int q  = qs[j];
            const int n0 = __byte_perm(0x11100100, 0x11100100, q >> 0);
            const int n1 = __byte_perm(0x11100100, 0x11100100, q >> 2);
            const int s0 = __byte_perm(0x01FF, 0x01FF, n0 >>  0);
            const int s1 = __byte_perm(0x01FF, 0x01FF, n1 >>  0);
            const int s2 = __byte_perm(0x01FF, 0x01FF, n0 >> 16);
            const int s3 = __byte_perm(0x01FF, 0x01FF, n1 >> 16);
            v[4*j + 0] = __byte_perm(s0, s1, 0x5410);
            v[4*j + 1] = __byte_perm(s0, s1, 0x7632);
            v[4*j + 2] = __byte_perm(s2, s3, 0x5410);
            v[4*j + 3] = __byte_perm(s2, s3, 0x7632);
        }
    } else if constexpr (type == GGML_TYPE_Q2_0) {
        const block_q2_0 * b = (const block_q2_0 *) row + c / 2;
        d = __half2float(b->d);
        const int16_t * qs = (const int16_t *) b->qs + (c % 2) * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q  = qs[j];
            const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
            const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
            v[2*j + 0] = __byte_perm(qe, qo, 0x5140);
            v[2*j + 1] = __byte_perm(qe, qo, 0x7362);
        }
    } else if constexpr (type == GGML_TYPE_Q4_0) {
        const block_q4_0 * b = (const block_q4_0 *) row + c;
        d = __half2float(b->d);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = get_int_b2(b->qs, j);
            v[j + 0] = __vsub4((q >> 0) & 0x0F0F0F0F, 0x08080808); // values 0..15 - 8: no borrow between bytes
            v[j + 4] = __vsub4((q >> 4) & 0x0F0F0F0F, 0x08080808);
        }
    } else {
        static_assert(type == GGML_TYPE_Q8_0, "unsupported type");
        const block_q8_0 * b = (const block_q8_0 *) row + c;
        d = __half2float(b->d);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            v[j] = get_int_b2(b->qs, j);
        }
    }
}

// token range of the grouped path
#define MMID_GROUPED_MIN_TOKENS 4
#define MMID_GROUPED_MAX_TOKENS 8192

struct mmid_grouped_seg {
    int expert; // routed: global id; shared: n_routed + j; -1: skipped pairs (output 0)
    int p0;     // first pair in pairs[]
    int n;      // pairs in the segment
};

// One block. Sorts the pairs (token t, slot s) -> t * n_slots + s by expert and splits the experts into segments.
// Shared expert j (slot n_used + j) takes every token. Shared segments are emitted first: they are the most work.
// Pairs whose id is outside the routed experts (skipped, pruned) go to a last, null expert, whose output is 0.
static __global__ void mmid_grouped_sort(const int32_t * __restrict__ ids, const int64_t ids_s1, const int n_used,
        const int n_sh, const int n_tokens, const int n_routed, int * __restrict__ pairs,
        mmid_grouped_seg * __restrict__ segs, int * __restrict__ n_segs) {
    __shared__ int count[MMID_GROUPED_MAX_E];
    __shared__ int offs[MMID_GROUPED_MAX_E];
    __shared__ int seg_offs[MMID_GROUPED_MAX_E];
    __shared__ int warp_sums[2][32];

    const int n_e    = n_routed + n_sh + 1; // + null expert
    const int e_null = n_routed + n_sh;
    const int tid    = threadIdx.x;
    const int nth    = blockDim.x;
    const int n_rout = n_used * n_tokens;

    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();

    for (int e = tid; e < n_e; e += nth) {
        count[e] = e < n_routed || e == e_null ? 0 : n_tokens;
    }
    __syncthreads();
    for (int i = tid; i < n_rout; i += nth) {
        const int id = ids[i % n_used + (i / n_used) * ids_s1];
        atomicAdd(&count[id >= 0 && id < n_routed ? id : e_null], 1);
    }
    __syncthreads();

    // exclusive scans of the pair counts and the segment counts; experts in the order shared, then routed
    const int per_thread = (n_e + nth - 1) / nth;
    int sum_p = 0;
    int sum_s = 0;
    for (int k = 0; k < per_thread; ++k) {
        const int i = tid * per_thread + k;
        if (i < n_e) {
            const int e = i < n_sh ? n_routed + i : i < n_sh + n_routed ? i - n_sh : e_null;
            sum_p += count[e];
            sum_s += (count[e] + MMID_GROUPED_SEG - 1) / MMID_GROUPED_SEG;
        }
    }
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    int incl_p = sum_p;
    int incl_s = sum_s;
#pragma unroll
    for (int o = 1; o < WARP_SIZE; o *= 2) {
        const int tp = __shfl_up_sync(0xFFFFFFFF, incl_p, o);
        const int ts = __shfl_up_sync(0xFFFFFFFF, incl_s, o);
        if (lane >= o) {
            incl_p += tp;
            incl_s += ts;
        }
    }
    if (lane == WARP_SIZE - 1) {
        warp_sums[0][warp] = incl_p;
        warp_sums[1][warp] = incl_s;
    }
    __syncthreads();
    if (warp == 0) {
        const int nw = nth / WARP_SIZE;
        int wp = lane < nw ? warp_sums[0][lane] : 0;
        int ws = lane < nw ? warp_sums[1][lane] : 0;
#pragma unroll
        for (int o = 1; o < WARP_SIZE; o *= 2) {
            const int tp = __shfl_up_sync(0xFFFFFFFF, wp, o);
            const int ts = __shfl_up_sync(0xFFFFFFFF, ws, o);
            if (lane >= o) {
                wp += tp;
                ws += ts;
            }
        }
        if (lane < nw) {
            warp_sums[0][lane] = wp;
            warp_sums[1][lane] = ws;
        }
    }
    __syncthreads();
    int run_p = incl_p - sum_p + (warp > 0 ? warp_sums[0][warp - 1] : 0);
    int run_s = incl_s - sum_s + (warp > 0 ? warp_sums[1][warp - 1] : 0);
    for (int k = 0; k < per_thread; ++k) {
        const int i = tid * per_thread + k;
        if (i < n_e) {
            const int e = i < n_sh ? n_routed + i : i < n_sh + n_routed ? i - n_sh : e_null;
            offs[e]     = run_p;
            seg_offs[e] = run_s;
            run_p += count[e];
            run_s += (count[e] + MMID_GROUPED_SEG - 1) / MMID_GROUPED_SEG;
        }
    }
    if (tid == nth - 1) {
        *n_segs = run_s;
    }
    __syncthreads();

    // segments
    for (int e = tid; e < n_e; e += nth) {
        const int c = count[e];
        for (int k = 0; k * MMID_GROUPED_SEG < c; ++k) {
            segs[seg_offs[e] + k] = { e == e_null ? -1 : e, offs[e] + k * MMID_GROUPED_SEG, min(MMID_GROUPED_SEG, c - k * MMID_GROUPED_SEG) };
        }
    }
    // pairs: shared experts in token order, routed through a cursor per expert
    const int n_slots = n_used + n_sh;
    for (int i = tid; i < n_sh * n_tokens; i += nth) {
        const int j = i / n_tokens;
        const int t = i % n_tokens;
        pairs[offs[n_routed + j] + t] = t * n_slots + n_used + j;
    }
    for (int i = tid; i < n_rout; i += nth) {
        const int s  = i % n_used;
        const int t  = i / n_used;
        const int id = ids[s + t * ids_s1];
        pairs[atomicAdd(&offs[id >= 0 && id < n_routed ? id : e_null], 1)] = t * n_slots + s;
    }
}

// grid: (row tiles, max segments); block: (WARP_SIZE, MMID_GROUPED_NWARPS). lpr lanes share one output row (one
// gate/up row pair for swiglu) and split its chunks. Output for pair (t, s), row r: dst[r + s * dst_s1 + t * dst_s2].
template <ggml_type type, bool swiglu, int lpr>
static __device__ __forceinline__ void mmid_grouped_body(const char * __restrict__ xe, const int64_t nb1,
        const block_q8_1 * __restrict__ y, const int64_t y_row_blocks, const bool y_bcast,
        const int * __restrict__ pairs, const int p0, const int n_pairs, const int n_slots,
        const int nchunks, const int row, const bool valid, const int rows_out, const int li,
        float * __restrict__ dst, const int64_t dst_s1, const int64_t dst_s2) {
    const char * xrow    = xe + row * nb1;
    const char * xrow_up = xe + (row + rows_out) * nb1;

    for (int pb = 0; pb < n_pairs; pb += MMID_GROUPED_NT) {
        const int nt = min(MMID_GROUPED_NT, n_pairs - pb);
        int ybase[MMID_GROUPED_NT];
        int pr[MMID_GROUPED_NT];
#pragma unroll
        for (int j = 0; j < MMID_GROUPED_NT; ++j) {
            pr[j]    = pairs[p0 + pb + min(j, nt - 1)];
            ybase[j] = (y_bcast ? pr[j] / n_slots : pr[j]) * y_row_blocks;
        }

        float acc[MMID_GROUPED_NT]    = {0.0f};
        float acc_up[MMID_GROUPED_NT] = {0.0f};
        for (int c = li; c < nchunks; c += lpr) {
            int   v[8];
            int   vu[8];
            float d;
            float du = 0.0f;
            mmid_grouped_load_chunk<type>(xrow, c, v, d);
            if constexpr (swiglu) {
                mmid_grouped_load_chunk<type>(xrow_up, c, vu, du);
            }
#pragma unroll
            for (int j = 0; j < MMID_GROUPED_NT; ++j) {
                if (j < nt) {
                    const block_q8_1 * yb = y + ybase[j] + c;
                    const float dy = __low2float(yb->ds);
                    int u[8];
#pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        u[k] = get_int_b4(yb->qs, k);
                    }
                    int si = 0;
#pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        si = ggml_cuda_dp4a(v[k], u[k], si);
                    }
                    acc[j] += d * dy * (float) si;
                    if constexpr (swiglu) {
                        int su = 0;
#pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            su = ggml_cuda_dp4a(vu[k], u[k], su);
                        }
                        acc_up[j] += du * dy * (float) su;
                    }
                }
            }
        }
#pragma unroll
        for (int j = 0; j < MMID_GROUPED_NT; ++j) {
#pragma unroll
            for (int o = lpr / 2; o > 0; o >>= 1) {
                acc[j] += __shfl_xor_sync(0xFFFFFFFF, acc[j], o, lpr);
                if constexpr (swiglu) {
                    acc_up[j] += __shfl_xor_sync(0xFFFFFFFF, acc_up[j], o, lpr);
                }
            }
        }
        if (li == 0 && valid) {
#pragma unroll
            for (int j = 0; j < MMID_GROUPED_NT; ++j) {
                if (j < nt) {
                    const int t = pr[j] / n_slots;
                    const int s = pr[j] % n_slots;
                    dst[row + s * dst_s1 + t * dst_s2] = swiglu ? ggml_cuda_op_silu_single(acc[j]) * acc_up[j] : acc[j];
                }
            }
        }
    }
}

template <bool swiglu, int lpr>
static __global__ void mul_mat_id_multi_grouped(const mmid_multi_experts experts, const block_q8_1 * __restrict__ y,
        const int64_t y_row_blocks, const bool y_bcast, const int * __restrict__ pairs,
        const mmid_grouped_seg * __restrict__ segs, const int * __restrict__ n_segs, const int n_slots,
        const int ncols, const int rows_out, float * __restrict__ dst, const int64_t dst_s1, const int64_t dst_s2) {
    constexpr int rows_per_warp  = WARP_SIZE / lpr;
    constexpr int rows_per_block = MMID_GROUPED_NWARPS * rows_per_warp;

    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();

    if ((int) blockIdx.y >= *n_segs) {
        return;
    }
    const mmid_grouped_seg seg = segs[blockIdx.y];
    if (seg.expert < 0) {
        const int row = blockIdx.x * rows_per_block + threadIdx.y * rows_per_warp + threadIdx.x / lpr;
        if (threadIdx.x % lpr == 0 && row < rows_out) {
            for (int j = 0; j < seg.n; ++j) {
                const int p = pairs[seg.p0 + j];
                dst[row + (p % n_slots) * dst_s1 + (p / n_slots) * dst_s2] = 0.0f;
            }
        }
        return;
    }
    const int n_routed = experts.first[experts.n_as];
    int g;
    int e;
    if (seg.expert >= n_routed) {
        g = experts.n_as;
        e = seg.expert - n_routed;
    } else {
        g = 0;
        while (g + 1 < experts.n_as && seg.expert >= experts.first[g + 1]) {
            ++g;
        }
        e = seg.expert - experts.first[g];
    }
    const char *  xe  = experts.data[g] + (int64_t) e * experts.nb2[g];
    const int64_t nb1 = experts.nb1[g];

    const int li  = threadIdx.x % lpr;
    const int row = blockIdx.x * rows_per_block + threadIdx.y * rows_per_warp + threadIdx.x / lpr;
    // rows past the end compute on row 0 (in bounds) and do not write: the shuffles need every lane
    const int row_c   = row < rows_out ? row : 0;
    const int nchunks = ncols / QK8_1;

#define MMID_GROUPED_BODY(T) mmid_grouped_body<T, swiglu, lpr>(xe, nb1, y, y_row_blocks, y_bcast, pairs, seg.p0, seg.n, \
        n_slots, nchunks, row_c, row < rows_out, rows_out, li, dst, dst_s1, dst_s2); break
    switch (experts.type[g]) {
        case GGML_TYPE_Q1_0: MMID_GROUPED_BODY(GGML_TYPE_Q1_0);
        case GGML_TYPE_Q2_0: MMID_GROUPED_BODY(GGML_TYPE_Q2_0);
        case GGML_TYPE_Q4_0: MMID_GROUPED_BODY(GGML_TYPE_Q4_0);
        default:             MMID_GROUPED_BODY(GGML_TYPE_Q8_0);
    }
#undef MMID_GROUPED_BODY
}

// ---------------------------------------------------------------------------------------------------------------------
// Grouped path on tensor cores (Ampere+): the same segments, with int8 mma.m16n8k32. A warp computes 16 rows (16 gate
// and 16 up rows for swiglu) x up to MMID_GROUPED_SEG tokens (n8 tiles). For every chunk of 32 values, thread
// q = lane % 4 of the A fragment needs the values k = 4q..4q+3 and 16+4q..16+4q+3 of its two rows: for Q1_0, Q2_0,
// Q4_0 and Q8_0 these are directly addressable in the packed blocks, so the activations keep the plain q8_1 layout.
// The int32 result of each chunk is scaled by the row's and the token's scales and accumulated in F32.

#define MMID_MMA_NWARPS 4
#ifndef MMID_MMA_CB
#define MMID_MMA_CB     1 // chunks whose weights are loaded ahead of their mmas
#endif

// 4 bits -> 4 bytes of +-1 (bit set: +1)
static __device__ __forceinline__ int mmid_mma_spread1(const int x) {
    const int m = (x * 0x00204081) & 0x01010101;
    return __vsub4(m << 1, 0x01010101);
}

// 4 crumbs -> 4 bytes of crumb - 1 (Q2_0: -1, 0, 1, 2)
static __device__ __forceinline__ int mmid_mma_spread2(const int b) {
    const int y = (b & 0x03) | ((b & 0x0C) << 6) | ((b & 0x30) << 12) | ((b & 0xC0) << 18);
    return __vsub4(y, 0x01010101);
}

template <ggml_type type>
static __device__ __forceinline__ void mmid_mma_load_a(const char * __restrict__ row, const int c, const int q,
        int & lo, int & hi, float & d) {
    if constexpr (type == GGML_TYPE_Q1_0) {
        const block_q1_0 * b  = (const block_q1_0 *) row + c / 4;
        const uint8_t    * qs = b->qs + 4 * (c % 4);
        const int          sh = 4 * (q & 1);
        d  = __half2float(b->d);
        lo = mmid_mma_spread1((qs[(q >> 1) + 0] >> sh) & 0xF);
        hi = mmid_mma_spread1((qs[(q >> 1) + 2] >> sh) & 0xF);
    } else if constexpr (type == GGML_TYPE_Q2_0) {
        const block_q2_0 * b  = (const block_q2_0 *) row + c / 2;
        const uint8_t    * qs = b->qs + 8 * (c % 2);
        d  = __half2float(b->d);
        lo = mmid_mma_spread2(qs[q + 0]);
        hi = mmid_mma_spread2(qs[q + 4]);
    } else if constexpr (type == GGML_TYPE_Q4_0) {
        const block_q4_0 * b = (const block_q4_0 *) row + c;
        const int v = get_int_b2(b->qs, q);
        d  = __half2float(b->d);
        lo = __vsub4((v >> 0) & 0x0F0F0F0F, 0x08080808);
        hi = __vsub4((v >> 4) & 0x0F0F0F0F, 0x08080808);
    } else {
        static_assert(type == GGML_TYPE_Q8_0, "unsupported type");
        const block_q8_0 * b = (const block_q8_0 *) row + c;
        d  = __half2float(b->d);
        lo = get_int_b2(b->qs, q + 0);
        hi = get_int_b2(b->qs, q + 4);
    }
}

static __device__ __forceinline__ void mmid_mma_16832(int (&c)[4], const int a0, const int a1, const int a2, const int a3,
        const int b0, const int b1) {
#ifdef AMPERE_MMA_AVAILABLE
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(c, a0, a1, a2, a3, b0, b1);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

constexpr int MMID_MMA_NT = MMID_GROUPED_SEG / 8; // n8 tiles per segment

// partial sums over the chunks c0, c0 + cstep, ... of rows row0 + gid and row0 + gid + 8 (clamped) for the segment's
// pairs (at most MMID_GROUPED_SEG): acc[t][i] is the C fragment of tile t (accu: the up rows for swiglu)
template <ggml_type type, bool swiglu>
static __device__ __forceinline__ void mmid_mma_body(const char * __restrict__ xe, const int64_t nb1,
        const block_q8_1 * __restrict__ y, const int64_t y_row_blocks, const bool y_bcast,
        const int * __restrict__ pairs, const int np, const int n_slots, const int nchunks, const int c0, const int cstep,
        const int row0, const int rows_out, float (&acc)[MMID_MMA_NT][4], float (&accu)[MMID_MMA_NT][4]) {
    const int lane = threadIdx.x;
    const int gid  = lane >> 2;
    const int q    = lane & 3;
    const int ntiles = (np + 7) / 8;

    const int ra = min(row0 + gid,     rows_out - 1);
    const int rb = min(row0 + gid + 8, rows_out - 1);
    const char * xa  = xe + ra * nb1;
    const char * xb  = xe + rb * nb1;
    const char * xau = xe + (ra + rows_out) * nb1;
    const char * xbu = xe + (rb + rows_out) * nb1;

    int yb[MMID_MMA_NT];    // B fragment: token gid of each tile
    int yc[MMID_MMA_NT][2]; // C fragment: tokens 2q, 2q + 1 of each tile
#pragma unroll
    for (int t = 0; t < MMID_MMA_NT; ++t) {
        const int pbx = pairs[min(8*t + gid,     np - 1)];
        const int pc0 = pairs[min(8*t + 2*q,     np - 1)];
        const int pc1 = pairs[min(8*t + 2*q + 1, np - 1)];
        yb[t]    = (y_bcast ? pbx / n_slots : pbx) * y_row_blocks;
        yc[t][0] = (y_bcast ? pc0 / n_slots : pc0) * y_row_blocks;
        yc[t][1] = (y_bcast ? pc1 / n_slots : pc1) * y_row_blocks;
    }

    // MMID_MMA_CB chunks at a time: all their weight fragments are loaded before the mmas, so the loads overlap
    for (int cb = c0; cb < nchunks; cb += MMID_MMA_CB * cstep) {
        int   a[MMID_MMA_CB][4];
        int   u[MMID_MMA_CB][4];
        float da[MMID_MMA_CB][2];
        float du[MMID_MMA_CB][2];
#pragma unroll
        for (int i = 0; i < MMID_MMA_CB; ++i) {
            const int c = min(cb + i * cstep, nchunks - 1);
            mmid_mma_load_a<type>(xa, c, q, a[i][0], a[i][2], da[i][0]);
            mmid_mma_load_a<type>(xb, c, q, a[i][1], a[i][3], da[i][1]);
            if constexpr (swiglu) {
                mmid_mma_load_a<type>(xau, c, q, u[i][0], u[i][2], du[i][0]);
                mmid_mma_load_a<type>(xbu, c, q, u[i][1], u[i][3], du[i][1]);
            }
        }
#pragma unroll
        for (int i = 0; i < MMID_MMA_CB; ++i) {
            const int c = cb + i * cstep;
            if (c >= nchunks) {
                break;
            }
#pragma unroll
            for (int t = 0; t < MMID_MMA_NT; ++t) {
                if (t < ntiles) {
                    const block_q8_1 * yy = y + yb[t] + c;
                    const int b0 = get_int_b4(yy->qs, q + 0);
                    const int b1 = get_int_b4(yy->qs, q + 4);
                    const float d0 = __low2float(y[yc[t][0] + c].ds);
                    const float d1 = __low2float(y[yc[t][1] + c].ds);
                    int cc[4] = {0, 0, 0, 0};
                    mmid_mma_16832(cc, a[i][0], a[i][1], a[i][2], a[i][3], b0, b1);
                    acc[t][0] += da[i][0] * d0 * (float) cc[0];
                    acc[t][1] += da[i][0] * d1 * (float) cc[1];
                    acc[t][2] += da[i][1] * d0 * (float) cc[2];
                    acc[t][3] += da[i][1] * d1 * (float) cc[3];
                    if constexpr (swiglu) {
                        int cu[4] = {0, 0, 0, 0};
                        mmid_mma_16832(cu, u[i][0], u[i][1], u[i][2], u[i][3], b0, b1);
                        accu[t][0] += du[i][0] * d0 * (float) cu[0];
                        accu[t][1] += du[i][0] * d1 * (float) cu[1];
                        accu[t][2] += du[i][1] * d0 * (float) cu[2];
                        accu[t][3] += du[i][1] * d1 * (float) cu[3];
                    }
                }
            }
        }
    }
}

// block: MMID_MMA_NWARPS warps = (MMID_MMA_NWARPS / ksplit) row groups of 16 rows x ksplit slices of the chunks;
// the slices are summed through shared memory
template <bool swiglu, int ksplit>
static __global__ void __launch_bounds__(WARP_SIZE * MMID_MMA_NWARPS) mul_mat_id_multi_mma(
        const mmid_multi_experts experts, const block_q8_1 * __restrict__ y,
        const int64_t y_row_blocks, const bool y_bcast, const int * __restrict__ pairs,
        const mmid_grouped_seg * __restrict__ segs, const int * __restrict__ n_segs, const int n_slots,
        const int ncols, const int rows_out, float * __restrict__ dst, const int64_t dst_s1, const int64_t dst_s2) {
    constexpr int rgroups        = MMID_MMA_NWARPS / ksplit;
    constexpr int rows_per_block = 16 * rgroups;
    constexpr int nacc           = MMID_MMA_NT * 4;

    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();

    if ((int) blockIdx.y >= *n_segs) {
        return;
    }
    const mmid_grouped_seg seg = segs[blockIdx.y];
    const int ks   = threadIdx.y % ksplit;
    const int row0 = blockIdx.x * rows_per_block + (threadIdx.y / ksplit) * 16;
    const int * sp = pairs + seg.p0;
    if (seg.expert < 0) {
        if (ks == 0) {
            for (int i = threadIdx.x; i < 16 * seg.n; i += WARP_SIZE) {
                const int row = row0 + i % 16;
                const int p   = sp[i / 16];
                if (row < rows_out) {
                    dst[row + (p % n_slots) * dst_s1 + (p / n_slots) * dst_s2] = 0.0f;
                }
            }
        }
        return;
    }
    const int n_routed = experts.first[experts.n_as];
    int g;
    int e;
    if (seg.expert >= n_routed) {
        g = experts.n_as;
        e = seg.expert - n_routed;
    } else {
        g = 0;
        while (g + 1 < experts.n_as && seg.expert >= experts.first[g + 1]) {
            ++g;
        }
        e = seg.expert - experts.first[g];
    }
    const char *  xe  = experts.data[g] + (int64_t) e * experts.nb2[g];
    const int64_t nb1 = experts.nb1[g];
    const int nchunks = ncols / QK8_1;

    float acc[MMID_MMA_NT][4]  = {};
    float accu[MMID_MMA_NT][4] = {};
#define MMID_MMA_BODY(T) mmid_mma_body<T, swiglu>(xe, nb1, y, y_row_blocks, y_bcast, sp, seg.n, n_slots, nchunks, ks, ksplit, \
        row0, rows_out, acc, accu); break
    switch (experts.type[g]) {
        case GGML_TYPE_Q1_0: MMID_MMA_BODY(GGML_TYPE_Q1_0);
        case GGML_TYPE_Q2_0: MMID_MMA_BODY(GGML_TYPE_Q2_0);
        case GGML_TYPE_Q4_0: MMID_MMA_BODY(GGML_TYPE_Q4_0);
        default:             MMID_MMA_BODY(GGML_TYPE_Q8_0);
    }
#undef MMID_MMA_BODY

    if constexpr (ksplit > 1) {
        __shared__ float red[MMID_MMA_NWARPS][swiglu ? 2 : 1][nacc][WARP_SIZE];
        if (ks > 0) {
#pragma unroll
            for (int i = 0; i < nacc; ++i) {
                red[threadIdx.y][0][i][threadIdx.x] = acc[i / 4][i % 4];
                if constexpr (swiglu) {
                    red[threadIdx.y][1][i][threadIdx.x] = accu[i / 4][i % 4];
                }
            }
        }
        __syncthreads();
        if (ks > 0) {
            return;
        }
#pragma unroll
        for (int k = 1; k < ksplit; ++k) {
#pragma unroll
            for (int i = 0; i < nacc; ++i) {
                acc[i / 4][i % 4] += red[threadIdx.y + k][0][i][threadIdx.x];
                if constexpr (swiglu) {
                    accu[i / 4][i % 4] += red[threadIdx.y + k][1][i][threadIdx.x];
                }
            }
        }
    }

    const int gid = threadIdx.x >> 2;
    const int q   = threadIdx.x & 3;
#pragma unroll
    for (int t = 0; t < MMID_MMA_NT; ++t) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int idx = 8*t + 2*q + j;
            if (idx >= seg.n) {
                continue;
            }
            const int p  = sp[idx];
            const int tk = p / n_slots;
            const int s  = p % n_slots;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = row0 + gid + 8*h;
                if (row < rows_out) {
                    const float gv = acc[t][2*h + j];
                    dst[row + s * dst_s1 + tk * dst_s2] = swiglu ? ggml_cuda_op_silu_single(gv) * accu[t][2*h + j] : gv;
                }
            }
        }
    }
}

bool ggml_cuda_mul_mat_id_multi_supported(const ggml_tensor * op) {
    const int n_as = ggml_get_op_params_i32(op, 0);
    const ggml_tensor * b   = op->src[n_as + 0];
    const ggml_tensor * ids = op->src[n_as + 1];
    const ggml_tensor * w   = op->src[n_as + 2];
    const ggml_tensor * sh  = op->src[n_as + 3];
    if (sh && (!mmid_multi_type_supported(sh->type) || !ggml_is_contiguous(sh))) {
        return false;
    }
    if (b->type != GGML_TYPE_F32 || b->nb[0] != sizeof(float) || ids->nb[0] != sizeof(int32_t) || !ggml_is_contiguous(op)) {
        return false;
    }
    if (w && !ggml_is_contiguous(w)) {
        return false;
    }
    for (int g = 0; g < n_as; ++g) {
        const ggml_tensor * a = op->src[g];
        if (!mmid_multi_type_supported(a->type) || !ggml_is_contiguous(a) || a->ne[0] % ggml_blck_size(a->type) != 0) {
            return false;
        }
    }
    return true;
}

template <bool swiglu, bool sum_slots, int ksplit>
static void mmid_multi_vec_launch(const mmid_multi_experts & experts, const block_q8_1 * y, const int32_t * ids,
        const float * w, float * dst, const int ncols, const int rows_out, const int n_used, const int n_slots,
        const int n_tokens, const int64_t y_row_blocks, const bool y_bcast, const int64_t ids_s1, const int64_t dst_s1,
        const int64_t dst_s2, const float * res, const int64_t res_s1, cudaStream_t stream) {
    constexpr int rows_per_block = MMID_MULTI_NWARPS / ksplit;
    const dim3 block(WARP_SIZE, MMID_MULTI_NWARPS, 1);
    const dim3 grid((rows_out + rows_per_block - 1) / rows_per_block, sum_slots ? n_tokens : n_slots * n_tokens, 1);
    ggml_cuda_kernel_launch(mul_mat_id_multi_vec<swiglu, sum_slots, ksplit>, ggml_cuda_kernel_launch_params(grid, block, 0, stream),
        experts, y, ids, w, dst, ncols, rows_out, n_used, n_slots, y_row_blocks, y_bcast, ids_s1, dst_s1, dst_s2, res, res_s1);
}

bool ggml_cuda_mul_mat_id_multi_can_fuse_add(const ggml_tensor * node, const ggml_tensor * res, const ggml_tensor * add_dst) {
    const int n_as = ggml_get_op_params_i32(node, 0);
    const bool swiglu = ggml_get_op_params_i32(node, 1) != 0;
    if (swiglu || node->src[n_as + 2] == nullptr) {
        return false;
    }
    if (node->type != GGML_TYPE_F32 || res->type != GGML_TYPE_F32 || add_dst->type != GGML_TYPE_F32 ||
            !ggml_are_same_shape(res, add_dst) || !ggml_are_same_shape(node, add_dst) ||
            res->nb[0] != sizeof(float) || add_dst->nb[0] != sizeof(float) ||
            node->ne[2] != 1 || node->ne[3] != 1) {
        return false;
    }
    const auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * a0 = (const char *) a->data;
        const char * b0 = (const char *) b->data;
        return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
    };
    // the output must not overwrite an input of the op; the residual may be updated in place (same layout)
    for (int j = 0; j < GGML_MAX_SRC && node->src[j]; ++j) {
        if (overlap(add_dst, node->src[j])) {
            return false;
        }
    }
    return !overlap(add_dst, res) || (res->data == add_dst->data && res->nb[1] == add_dst->nb[1]);
}

void ggml_cuda_mul_mat_id_multi(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * res,
        ggml_tensor * add_dst) {
    const int  n_as   = ggml_get_op_params_i32(dst, 0);
    const bool swiglu = ggml_get_op_params_i32(dst, 1) != 0;

    const ggml_tensor * b   = dst->src[n_as + 0];
    const ggml_tensor * ids = dst->src[n_as + 1];
    const ggml_tensor * w   = dst->src[n_as + 2];
    const ggml_tensor * sh  = dst->src[n_as + 3];

    cudaStream_t stream = ctx.stream();

    const int64_t ncols    = b->ne[0];
    const int64_t rows     = dst->src[0]->ne[1];
    const int64_t rows_out = swiglu ? rows / 2 : rows;
    const int64_t n_used   = ids->ne[0];
    const int64_t n_slots  = n_used + (sh ? sh->ne[2] : 0);
    const int64_t n_tokens = ids->ne[1];
    const int64_t ids_s1   = ids->nb[1] / sizeof(int32_t);

    // fused residual add: the weighted sum over slots plus res goes to add_dst, dst is not written
    GGML_ASSERT(!res || (w && !swiglu && add_dst));
    float * const   dst_d  = (float *) (res ? add_dst->data : dst->data);
    const int64_t   dst_s1 = (res ? add_dst->nb[1] : dst->nb[1]) / sizeof(float);
    const float *   res_d  = res ? (const float *) res->data : nullptr;
    const int64_t   res_s1 = res ? res->nb[1] / sizeof(float) : 0;

    // all tensors, the shared one last
    const ggml_tensor * as[GGML_MUL_MAT_ID_MULTI_MAX + 1];
    const int n_tensors = n_as + (sh ? 1 : 0);
    for (int g = 0; g < n_as; ++g) {
        as[g] = dst->src[g];
    }
    if (sh) {
        as[n_as] = sh;
    }

    mmid_multi_experts experts = {};
    experts.n_as = n_as;
    for (int g = 0; g < n_tensors; ++g) {
        experts.data[g] = (const char *) as[g]->data;
        experts.nb1[g]  = as[g]->nb[1];
        experts.nb2[g]  = as[g]->nb[2];
        experts.type[g] = as[g]->type;
    }
    for (int g = 0; g < n_as; ++g) {
        experts.first[g + 1] = experts.first[g] + (int) as[g]->ne[2];
    }

    // Q4_1/Q5_0/Q5_1 dot products use the F16 block sum of q8_1 (overflow, see above): MMQ for those
    bool vec_ok = true;
    for (int g = 0; g < n_tensors; ++g) {
        const ggml_type t = as[g]->type;
        vec_ok = vec_ok && (t == GGML_TYPE_Q1_0 || t == GGML_TYPE_Q2_0 || t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q8_0);
    }

    const int n_routed = experts.first[n_as];
    const int n_sh     = sh ? (int) sh->ne[2] : 0;
    if (vec_ok && n_tokens >= MMID_GROUPED_MIN_TOKENS && n_tokens <= MMID_GROUPED_MAX_TOKENS &&
            n_routed + n_sh + 1 <= MMID_GROUPED_MAX_E && ncols % QK8_1 == 0) {
        const int64_t ne10_padded  = GGML_PAD(ncols, MATRIX_ROW_PADDING);
        const int64_t y_row_blocks = ne10_padded / QK8_1;
        const int64_t n_rows_b     = b->ne[1] * b->ne[2];
        const void * y_cached = ncols == ne10_padded ? ctx.q8_1_lookup(b) : nullptr;
        ggml_cuda_pool_alloc<char> y(ctx.pool());
        if (!y_cached) {
            y.alloc(n_rows_b * y_row_blocks * sizeof(block_q8_1));
            quantize_row_q8_1_cuda((const float *) b->data, y.get(), ncols,
                b->nb[1] / sizeof(float), b->nb[2] / sizeof(float), b->nb[3] / sizeof(float),
                ne10_padded, b->ne[1], b->ne[2], 1, stream);
        }
        const block_q8_1 * yq = (const block_q8_1 *) (y_cached ? y_cached : y.get());

        const int64_t n_rout   = n_used * n_tokens;
        const int64_t n_pairs  = n_rout + n_sh * n_tokens;
        const int64_t max_segs = std::min<int64_t>(n_routed, n_rout) + n_rout / MMID_GROUPED_SEG +
                                 n_sh * ((n_tokens + MMID_GROUPED_SEG - 1) / MMID_GROUPED_SEG) + 2;
        ggml_cuda_pool_alloc<int>              pairs(ctx.pool(), n_pairs);
        ggml_cuda_pool_alloc<mmid_grouped_seg> segs(ctx.pool(), max_segs);
        ggml_cuda_pool_alloc<int>              n_segs(ctx.pool(), 1);
        int              * pairs_p  = pairs.get();
        mmid_grouped_seg * segs_p   = segs.get();
        int              * n_segs_p = n_segs.get();
        ggml_cuda_kernel_launch(mmid_grouped_sort, ggml_cuda_kernel_launch_params(dim3(1), dim3(1024), 0, stream),
            (const int32_t *) ids->data, ids_s1, (int) n_used, n_sh, (int) n_tokens, n_routed, pairs_p, segs_p, n_segs_p);

        // the weighted sum over slots goes through tmp [rows, n_slots, n_tokens]
        const bool weighted = !swiglu && w;
        ggml_cuda_pool_alloc<float> tmp(ctx.pool(), weighted ? rows * n_slots * n_tokens : 0);
        float * out = weighted ? tmp.get() : (float *) dst->data;
        const int64_t out_s1 = weighted ? rows : dst->nb[1] / sizeof(float);
        const int64_t out_s2 = weighted ? rows * n_slots : dst->nb[2] / sizeof(float);
        const bool y_bcast = b->ne[1] == 1;

        const int nchunks = ncols / QK8_1;
        const int lpr = nchunks >= 64 ? 16 : nchunks >= 32 ? 8 : 4; // about 4 chunks per lane
        const dim3 block(WARP_SIZE, MMID_GROUPED_NWARPS, 1);
#define MMID_GROUPED_LAUNCH(SWIGLU, LPR) do { \
            const int rpb = MMID_GROUPED_NWARPS * (WARP_SIZE / LPR); \
            const dim3 grid((rows_out + rpb - 1) / rpb, max_segs, 1); \
            ggml_cuda_kernel_launch(mul_mat_id_multi_grouped<SWIGLU, LPR>, ggml_cuda_kernel_launch_params(grid, block, 0, stream), \
                experts, yq, y_row_blocks, y_bcast, (const int *) pairs_p, (const mmid_grouped_seg *) segs_p, \
                (const int *) n_segs_p, (int) n_slots, (int) ncols, (int) rows_out, out, out_s1, out_s2); \
        } while (0)
        // tensor cores from this many tokens on: below it dp4a is faster (RTX 5090)
        const int mma_min = nchunks >= 64 ? 96 : 24;
        const bool use_mma = n_tokens >= mma_min && ampere_mma_available(ggml_cuda_info().devices[ctx.device].cc);
        if (use_mma) {
            const bool ksplit4 = nchunks >= 64;
            const dim3 block_mma(WARP_SIZE, MMID_MMA_NWARPS, 1);
#define MMID_MMA_LAUNCH(SWIGLU, KSPLIT) do { \
                const int rpb = 16 * (MMID_MMA_NWARPS / KSPLIT); \
                const dim3 grid((rows_out + rpb - 1) / rpb, max_segs, 1); \
                ggml_cuda_kernel_launch(mul_mat_id_multi_mma<SWIGLU, KSPLIT>, ggml_cuda_kernel_launch_params(grid, block_mma, 0, stream), \
                    experts, yq, y_row_blocks, y_bcast, (const int *) pairs_p, (const mmid_grouped_seg *) segs_p, \
                    (const int *) n_segs_p, (int) n_slots, (int) ncols, (int) rows_out, out, out_s1, out_s2); \
            } while (0)
            if (swiglu) {
                if (ksplit4) {
                    MMID_MMA_LAUNCH(true, 4);
                } else {
                    MMID_MMA_LAUNCH(true, 1);
                }
            } else {
                if (ksplit4) {
                    MMID_MMA_LAUNCH(false, 4);
                } else {
                    MMID_MMA_LAUNCH(false, 1);
                }
            }
#undef MMID_MMA_LAUNCH
        } else if (swiglu) {
            switch (lpr) {
                case 16: MMID_GROUPED_LAUNCH(true, 16); break;
                case 8:  MMID_GROUPED_LAUNCH(true, 8);  break;
                default: MMID_GROUPED_LAUNCH(true, 4);  break;
            }
        } else {
            switch (lpr) {
                case 16: MMID_GROUPED_LAUNCH(false, 16); break;
                case 8:  MMID_GROUPED_LAUNCH(false, 8);  break;
                default: MMID_GROUPED_LAUNCH(false, 4);  break;
            }
        }
#undef MMID_GROUPED_LAUNCH
        if (weighted) {
            const int64_t n = rows * n_tokens;
            mmid_multi_weighted_sum<<<(n + 255) / 256, 256, 0, stream>>>(tmp.get(), (const float *) w->data,
                dst_d, rows, n_used, n_slots, n_tokens, dst_s1, res_d, res_s1);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (vec_ok && n_tokens <= MMID_MULTI_MAX_BATCH) {
        // quantize b to q8_1 once for all tensors (every supported type takes q8_1 activations)
        const int64_t ne10_padded  = GGML_PAD(ncols, MATRIX_ROW_PADDING);
        const int64_t y_row_blocks = ne10_padded / QK8_1;
        const int64_t n_rows_b     = b->ne[1] * b->ne[2];
        const void * y_cached = ncols == ne10_padded ? ctx.q8_1_lookup(b) : nullptr;
        ggml_cuda_pool_alloc<char> y(ctx.pool());
        if (!y_cached) {
            y.alloc(n_rows_b * y_row_blocks * sizeof(block_q8_1));
            quantize_row_q8_1_cuda((const float *) b->data, y.get(), ncols,
                b->nb[1] / sizeof(float), b->nb[2] / sizeof(float), b->nb[3] / sizeof(float),
                ne10_padded, b->ne[1], b->ne[2], 1, stream);
        }

        const int64_t dst_s2 = dst->nb[2] / sizeof(float);
        const block_q8_1 * yq = (const block_q8_1 *) (y_cached ? y_cached : y.get());
        const bool y_bcast = b->ne[1] == 1;
        const float * wd = w ? (const float *) w->data : nullptr;
        float * dd = dst_d;
        const int32_t * idd = (const int32_t *) ids->data;
#define MMID_MULTI_LAUNCH(SWIGLU, SUM, KSPLIT) mmid_multi_vec_launch<SWIGLU, SUM, KSPLIT>(experts, yq, idd, wd, dd, \
            ncols, rows_out, n_used, n_slots, n_tokens, y_row_blocks, y_bcast, ids_s1, dst_s1, dst_s2, res_d, res_s1, stream)
        // two warps per row: faster than 1 or 4 on RTX 5090 for the Flint shapes
        if (swiglu) {
            MMID_MULTI_LAUNCH(true, false, 2);
        } else if (w) {
            MMID_MULTI_LAUNCH(false, true, 2);
        } else {
            MMID_MULTI_LAUNCH(false, false, 1);
        }
#undef MMID_MULTI_LAUNCH
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // MMQ per tensor into tmp [rows, n_slots, n_tokens], each on the slots of its experts; rows no tensor writes stay 0
    const bool direct = !swiglu && !w;
    ggml_cuda_pool_alloc<float>   tmp(ctx.pool(), direct ? 0 : rows * n_slots * n_tokens);
    ggml_cuda_pool_alloc<int32_t> ids_local(ctx.pool(), n_slots * n_tokens);

    ggml_tensor tmp_t = {};
    tmp_t.type  = GGML_TYPE_F32;
    tmp_t.data  = direct ? dst->data : (void *) tmp.get();
    tmp_t.ne[0] = rows; tmp_t.ne[1] = n_slots; tmp_t.ne[2] = n_tokens; tmp_t.ne[3] = 1;
    tmp_t.nb[0] = sizeof(float);
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        tmp_t.nb[i] = tmp_t.nb[i - 1] * tmp_t.ne[i - 1];
    }

    ggml_tensor ids_t = {};
    ids_t.type  = GGML_TYPE_I32;
    ids_t.data  = ids_local.get();
    ids_t.ne[0] = n_slots; ids_t.ne[1] = n_tokens; ids_t.ne[2] = 1; ids_t.ne[3] = 1;
    ids_t.nb[0] = sizeof(int32_t);
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        ids_t.nb[i] = ids_t.nb[i - 1] * ids_t.ne[i - 1];
    }

    const int64_t n_all = n_slots * n_tokens;
    CUDA_CHECK(cudaMemsetAsync(tmp_t.data, 0, ggml_nbytes(&tmp_t), stream));
    for (int g = 0; g < n_tensors; ++g) {
        const bool shared = g == n_as;
        mmid_multi_local_ids<<<(n_all + 255) / 256, 256, 0, stream>>>((const int32_t *) ids->data, ids_local.get(),
            n_used, n_slots, ids_s1, n_tokens, shared ? 0 : experts.first[g], shared ? 0 : experts.first[g + 1], shared);
        CUDA_CHECK(cudaGetLastError());
        ggml_cuda_mul_mat_q(ctx, as[g], b, &ids_t, &tmp_t, nullptr, /*skipped_slots =*/ true);
    }

    if (swiglu) {
        const int64_t n = rows_out * n_all;
        mmid_multi_swiglu<<<(n + 255) / 256, 256, 0, stream>>>(tmp.get(), (float *) dst->data, rows_out, n_all,
            dst->nb[1] / sizeof(float));
    } else if (w) {
        const int64_t n = rows * n_tokens;
        mmid_multi_weighted_sum<<<(n + 255) / 256, 256, 0, stream>>>(tmp.get(), (const float *) w->data,
            dst_d, rows, n_used, n_slots, n_tokens, dst_s1, res_d, res_s1);
    }
    CUDA_CHECK(cudaGetLastError());
}
