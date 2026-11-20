#include "flint-fused.cuh"
#include "cp-async.cuh"
#include "vecdotq.cuh"
#include "quantize.cuh"
#include "unary.cuh"

// Router: FR_BLOCKS blocks of FR_WARPS warps. Every block normalizes the activations, then computes FR_ROWS router logits
// (BF16 weights, F32 activations, the weights are loaded before the PDL wait): warp w does the part of the reference warp w
// of every row, so that a warp reads only its eighth of the activations. The last block to finish (counter in
// scratch) runs the sigmoid + bias top-k (as topk_moe_cuda) on the logits and resets the counter. With defer, the top-k is
// left to the next kernel that reads the ids (see ggml_cuda_topk_deferred). Block c also writes the normalized activations of token c as F32 and q8_1.
// The results are bit-exact with rms_norm_mul_q8_1_f32 (256 threads), mul_mat_vec_f (256 threads) and topk_moe_cuda:
// each thread does the work of FR_NH threads of those kernels, and the 8 warp sums are added in the order of their final
// xor butterfly.

#define FR_NEXPERT FLINT_NEXPERT
#define FR_NCOLS   2048
#define FR_WARPS   8
#define FR_NH      (256 / (FR_WARPS * WARP_SIZE)) // reference threads per thread
#define FR_THREADS (FR_WARPS * WARP_SIZE)
#define FR_ROWS    FR_WARPS // router rows per block
#define FR_BLOCKS  (FR_NEXPERT / FR_ROWS)
#define FR_MAX_TOKENS 8 // more than FLINT_MAX_TOKENS: the unfused router ops are slow with more tokens

static_assert(FR_THREADS >= FLINT_NEXPERT, "the last block runs flint_topk_block");
static_assert(FR_BLOCKS >= FR_MAX_TOKENS, "block c writes the activations of token c");
static_assert(FR_MAX_TOKENS >= FLINT_MAX_TOKENS, "the router takes the tokens of the other fused kernels");
static_assert(FR_NCOLS == 2 * 4 * 256, "the reference kernels use 256 threads with 2 float4 or 4 bf16 pairs each");

// tokens of the fused kernels (rows of the activations)
static int64_t flint_ntok(const ggml_tensor * t) {
    return t->ne[2] == 1 && t->ne[3] == 1 ? t->ne[1] : 0;
}

struct flint_route_scratch {
    unsigned int counter;
    unsigned int done; // several tokens: blocks that finished their top-k
    float        logits[FR_MAX_TOKENS][FR_NEXPERT];
};

// the sums over the lanes of a warp of V values per lane (V a power of 2), each as warp_reduce_sum computes it, with
// fewer shuffles: at each butterfly step the values are halved (the lanes with the offset bit set keep the upper half) until
// one is left. Lane l gets the sum of value l / (WARP_SIZE / V).
template <int V>
static __device__ __forceinline__ float flint_warp_sum_scatter(float * v) {
    static_assert(V >= 1 && V <= WARP_SIZE && (V & (V - 1)) == 0, "V must be a power of 2 <= WARP_SIZE");
    const int lane = threadIdx.x % WARP_SIZE;
#pragma unroll
    for (int step = 0; (WARP_SIZE >> step) > 1; ++step) {
        const int off = WARP_SIZE / 2 >> step;
        const int n   = V / 2 >> step; // values kept
        if (n == 0) {
            v[0] += __shfl_xor_sync(0xFFFFFFFF, v[0], off);
            continue;
        }
        const bool hi = lane & off;
#pragma unroll
        for (int i = 0; i < n; ++i) {
            const float keep = hi ? v[i + n] : v[i];
            const float send = hi ? v[i] : v[i + n];
            v[i] = keep + __shfl_xor_sync(0xFFFFFFFF, send, off);
        }
    }
    return v[0];
}

// the smallest power of 2 >= n
static constexpr __host__ __device__ int flint_pow2_ceil(const int n) {
    return n <= 1 ? 1 : 2 * flint_pow2_ceil((n + 1) / 2);
}

// xor butterfly result of 8 values on lanes 0..7 (the other lanes 0)
static __device__ __forceinline__ float flint_sum8(const float * v) {
    return ((v[0] + v[4]) + (v[2] + v[6])) + ((v[1] + v[5]) + (v[3] + v[7]));
}

// NC tokens: token c is row c of x and xn (strides x_s1, xn_s1), its ids/weights are column c (strides ids_s1, w_s1),
// its q8_1 activations start at block c*FR_NCOLS/QK8_1. Each token gets the result of a single-token run. Dynamic shared
// memory: the normalized activations, NC*FR_NCOLS floats.
template <int NC>
static __global__ void __launch_bounds__(FR_THREADS) flint_route_kernel(
        const float * x, const int64_t x_s1, const float * norm_w, float * xn, const int64_t xn_s1, block_q8_1 * yq,
        const char * router, const int64_t router_nb1, const float * bias, flint_route_scratch * scratch,
        int32_t * ids, const int64_t ids_s1, float * weights, const int64_t w_s1, const int n_used, const float eps,
        const float clamp_val, const float scale_val, const bool with_norm, const bool defer) {
    extern __shared__ float4 fr_xs[];
    float4 (*xs)[FR_NCOLS / 4] = (float4 (*)[FR_NCOLS / 4]) fr_xs;
    __shared__ float  red[NC][8];
    __shared__ float  part[NC][FR_ROWS][FR_WARPS]; // the reference warp sums of the logits
    __shared__ bool   last;

    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;

    // reference thread h*FR_THREADS + tid: float4 h*FR_THREADS + tid + 256*k
    float4 m[FR_NH][2];
#pragma unroll
    for (int h = 0; h < FR_NH; ++h) {
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            m[h][k] = ((const float4 *) norm_w)[h * FR_THREADS + tid + 256 * k];
        }
    }
    // router rows of this block
    static_assert(FR_WARPS == 8, "a warp per reference warp of mul_mat_vec_f");
    nv_bfloat162 wv[FR_ROWS][4];
#pragma unroll
    for (int r = 0; r < FR_ROWS; ++r) {
        const nv_bfloat162 * w2 = (const nv_bfloat162 *) (router + (blockIdx.x * FR_ROWS + r) * router_nb1);
#pragma unroll
        for (int it = 0; it < 4; ++it) {
            wv[r][it] = w2[warp * WARP_SIZE + lane + it * 256];
        }
    }

    const float bias_v = defer ? 0.0f : flint_topk_bias(bias);

    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();

    // the sums of squares of the reference threads (value c*FR_NH + h: token c, reference thread h), reduced over the warp
    // all at once (each as alone)
    constexpr int NS  = NC * FR_NH;
    constexpr int NSV = flint_pow2_ceil(NS);
    float4 v[NC][FR_NH][2];
    float  ss[NSV];
#pragma unroll
    for (int k = 0; k < NSV; ++k) {
        ss[k] = 0.0f;
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
    for (int h = 0; h < FR_NH; ++h) {
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            v[c][h][k] = ((const float4 *) (x + c * x_s1))[h * FR_THREADS + tid + 256 * k];
            ss[c * FR_NH + h] += v[c][h][k].x*v[c][h][k].x + v[c][h][k].y*v[c][h][k].y + v[c][h][k].z*v[c][h][k].z + v[c][h][k].w*v[c][h][k].w;
        }
    }
    }
    {
        const float sum = flint_warp_sum_scatter<NSV>(ss);
        const int   k   = lane / (WARP_SIZE / NSV);
        if (lane % (WARP_SIZE / NSV) == 0 && k < NS) {
            red[k / FR_NH][(k % FR_NH) * FR_WARPS + warp] = sum;
        }
    }
    __syncthreads();

#pragma unroll
    for (int c = 0; c < NC; ++c) {
    const float scale = rsqrtf(flint_sum8(red[c]) / FR_NCOLS + eps);
#pragma unroll
    for (int h = 0; h < FR_NH; ++h) {
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int i = h * FR_THREADS + tid + 256 * k;
            float4 r = make_float4(scale*v[c][h][k].x, scale*v[c][h][k].y, scale*v[c][h][k].z, scale*v[c][h][k].w);
            r.x *= m[h][k].x;
            r.y *= m[h][k].y;
            r.z *= m[h][k].z;
            r.w *= m[h][k].w;
            xs[c][i] = r;
            if (blockIdx.x == c) {
                ((float4 *) (xn + c * xn_s1))[i] = r;

                // q8_1: 8 threads per block of 32 values (the layout of quantize_q8_1)
                float amax = fmaxf(fmaxf(fabsf(r.x), fabsf(r.y)), fmaxf(fabsf(r.z), fabsf(r.w)));
                amax = warp_reduce_max<8>(amax);
                const float d = amax / 127.0f;
                char4 q;
                q.x = amax == 0.0f ? 0 : roundf(r.x / d);
                q.y = amax == 0.0f ? 0 : roundf(r.y / d);
                q.z = amax == 0.0f ? 0 : roundf(r.z / d);
                q.w = amax == 0.0f ? 0 : roundf(r.w / d);
                int qsum = (int) q.x + (int) q.y + (int) q.z + (int) q.w;
#pragma unroll
                for (int offset = 4; offset > 0; offset >>= 1) {
                    qsum += __shfl_xor_sync(0xffffffff, qsum, offset, 8);
                }
                block_q8_1 & yb = yq[c * (FR_NCOLS / QK8_1) + i / 8];
                ((char4 *) yb.qs)[i % 8] = q;
                if (i % 8 == 0) {
                    yb.ds = make_half2(d, d * qsum);
                }
            }
        }
    }
    }
    __syncthreads();

    // mul_mat_vec_f: reference warp vw, lane l sums the pairs vw*32 + l + 256*it in order; warp vw does it for every token
    // and row (value c*FR_ROWS + r), then they are reduced NV at once (each as alone)
    constexpr int NA = NC * FR_ROWS;                              // values per lane
    constexpr int NV = NA <= 8 ? 8 : NA <= 16 ? 16 : WARP_SIZE;   // per reduction, padded to a power of 2
    constexpr int NR = (NA + NV - 1) / NV;                         // reductions
    float a[NR * NV];
#pragma unroll
    for (int k = 0; k < NR * NV; ++k) {
        a[k] = 0.0f;
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
    const float2 * y2 = (const float2 *) xs[c];
#pragma unroll
    for (int it = 0; it < 4; ++it) {
        const float2 tmpy = y2[warp * WARP_SIZE + lane + it * 256];
#pragma unroll
        for (int r = 0; r < FR_ROWS; ++r) {
            ggml_cuda_mad(a[c * FR_ROWS + r], __bfloat1622float2(wv[r][it]), tmpy);
        }
    }
    }
#pragma unroll
    for (int g = 0; g < NR; ++g) {
        const float sum = flint_warp_sum_scatter<NV>(a + g * NV);
        const int   k   = g * NV + lane / (WARP_SIZE / NV);
        if (lane % (WARP_SIZE / NV) == 0 && k < NA) {
            part[k / FR_ROWS][k % FR_ROWS][warp] = sum;
        }
    }
    __syncthreads();
    if (tid < NC * FR_ROWS) {
        scratch->logits[tid / FR_ROWS][blockIdx.x * FR_ROWS + tid % FR_ROWS] = flint_sum8(part[tid / FR_ROWS][tid % FR_ROWS]);
    }
    if (defer) {
        return;
    }

    __shared__ int   ids_s[FLINT_MAX_USED];
    __shared__ float w_s[FLINT_MAX_USED];
    __threadfence();
    __syncthreads();
    if constexpr (NC == 1) {
        // last block: all logits are written
        if (tid == 0) {
            last = atomicAdd(&scratch->counter, 1) == FR_BLOCKS - 1;
        }
        __syncthreads();
        if (!last) {
            return;
        }
        __threadfence();
        flint_topk_block(scratch->logits[0], bias_v, n_used, with_norm, clamp_val, scale_val, ids_s, w_s);
        if (tid == 0) {
            scratch->counter = 0;
        }
        if (tid < n_used) {
            ids[tid]     = ids_s[tid];
            weights[tid] = w_s[tid];
        }
    } else {
        // block c < NC waits for all logits and does the top-k of token c (all blocks of the small grid are resident or
        // will be, they do not wait for each other); the last one resets the counters
        static_assert(NC <= FR_BLOCKS, "a block per token");
        if (tid == 0) {
            atomicAdd(&scratch->counter, 1);
            if (blockIdx.x < NC) {
                while (atomicAdd(&scratch->counter, 0) < FR_BLOCKS) {
                }
            }
        }
        if (blockIdx.x >= NC) {
            return;
        }
        __syncthreads();
        __threadfence();
        const int c = blockIdx.x;
        flint_topk_block(scratch->logits[c], bias_v, n_used, with_norm, clamp_val, scale_val, ids_s, w_s);
        if (tid < n_used) {
            ids[c * ids_s1 + tid]   = ids_s[tid];
            weights[c * w_s1 + tid] = w_s[tid];
        }
        if (tid == 0 && atomicAdd(&scratch->done, 1) == NC - 1) {
            scratch->done    = 0;
            scratch->counter = 0;
        }
    }
}

// RMS_NORM -> MUL -> MUL_MAT (Q4_0, FR_NCOLS columns) of NC tokens: FN_WARPS warps, each 4 warps do FN_MM_RPG rows
// (FN_MM_RPG_N with several tokens). Each block normalizes and quantizes the activations: per token, 256 threads do the
// work of rms_norm_mul_q8_1_f32 (256 threads). The dot products are those of mul_mat_vec_q for ncols_dst 1 (4 warps per
// row; thread t of a group: q4_0 block t/2, ints 2*(t%2), 2*(t%2) + 1), for every token, so each token gets the result
// of a single-token run. Few large blocks: every block reads the activations. The weights are loaded before the PDL wait.
// With a gate matrix vg (glu): the SWIGLU of mul_mat_vec_q with the fused gate (dst = up * silu(gate), vx: up), FN_GLU_RPG
// rows per group (grid in one wave with 2 blocks per SM).
#define FN_WARPS   16
#define FN_MM_RPG   4
#define FN_MM_RPG_N 6 // several tokens: fewer blocks (every block normalizes and quantizes all tokens)
#define FN_GLU_RPG  5

template <int FN_RPG_T, bool glu, int NC>
static __global__ void __launch_bounds__(FN_WARPS * WARP_SIZE) flint_norm_mmvq_kernel(
        const float * x, const int64_t x_s1, const float * norm_w, const float eps, const char * vx, const char * vg,
        const int64_t stride_row, const int nrows, float * dst, const int64_t dst_s1) {
    constexpr int FN_RPG    = FN_RPG_T;
    constexpr int FN_GROUPS = FN_WARPS / 4;
    constexpr int FN_ROWS   = FN_RPG * FN_GROUPS;
    constexpr int NT        = FN_WARPS * WARP_SIZE / 256; // tokens normalized at the same time
    static_assert(FR_NCOLS / QK4_0 == 4 * WARP_SIZE / 2, "one q4_0 block per 2 threads of a group");
    static_assert(FN_WARPS * WARP_SIZE % 256 == 0, "256 threads normalize a token");
    constexpr int TC        = glu ? 1 : (NC < 16 / FN_RPG ? NC : 16 / FN_RPG); // tokens reduced at once (static shared memory: 16 rows x tokens)
    __shared__ block_q8_1 ys[NC][FR_NCOLS / QK8_1];
    __shared__ float      red[NC][8];
    __shared__ float      part[FN_GROUPS][4][TC][FN_RPG][WARP_SIZE];
    __shared__ float      part_g[glu ? FN_GROUPS : 1][4][TC][glu ? FN_RPG : 1][WARP_SIZE];

    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int grp  = warp / 4;
    const int gw   = warp % 4;
    const int row0 = blockIdx.x * FN_ROWS + grp * FN_RPG;
    const int kbx  = (tid % 128) / 2;
    const int kqs  = 2 * (tid % 2);
    const int nt   = tid % 256; // normalizing thread of token tid/256 + NT*j: float4 nt + 256*k
    const int nw   = nt / WARP_SIZE;

    ggml_cuda_pdl_lc();

    float4 m[2];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        m[k] = ((const float4 *) norm_w)[nt + 256 * k];
    }
    int   v[FN_RPG][2];
    float d4[FN_RPG];
    int   vgt[glu ? FN_RPG : 1][2];
    float dg4[glu ? FN_RPG : 1];
#pragma unroll
    for (int i = 0; i < FN_RPG; ++i) {
        if (row0 + i < nrows) {
            const block_q4_0 * b = (const block_q4_0 *) vx + (row0 + i) * stride_row + kbx;
            v[i][0] = get_int_b2(b->qs, kqs + 0);
            v[i][1] = get_int_b2(b->qs, kqs + 1);
            d4[i]   = b->d;
            if constexpr (glu) {
                const block_q4_0 * bg = (const block_q4_0 *) vg + (row0 + i) * stride_row + kbx;
                vgt[i][0] = get_int_b2(bg->qs, kqs + 0);
                vgt[i][1] = get_int_b2(bg->qs, kqs + 1);
                dg4[i]    = bg->d;
            }
        } else {
            v[i][0] = v[i][1] = 0;
            d4[i]   = 0.0f;
            if constexpr (glu) {
                vgt[i][0] = vgt[i][1] = 0;
                dg4[i]    = 0.0f;
            }
        }
    }

    ggml_cuda_pdl_sync();

    constexpr int NJ = (NC + NT - 1) / NT;
    float4 xv[NJ][2];
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
        const int c = tid / 256 + NT * j;
        if (NC % NT == 0 || c < NC) {
            float sum = 0.0f;
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                xv[j][k] = ((const float4 *) (x + c * x_s1))[nt + 256 * k];
                sum += xv[j][k].x*xv[j][k].x + xv[j][k].y*xv[j][k].y + xv[j][k].z*xv[j][k].z + xv[j][k].w*xv[j][k].w;
            }
            sum = warp_reduce_sum(sum);
            if (lane == 0) {
                red[c][nw] = sum;
            }
        }
    }
    __syncthreads();

#pragma unroll
    for (int j = 0; j < NJ; ++j) {
        const int c = tid / 256 + NT * j;
        if (NC % NT != 0 && c >= NC) {
            continue;
        }
        const float scale = rsqrtf(flint_sum8(red[c]) / FR_NCOLS + eps);
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int i = nt + 256 * k;
            float4 r = make_float4(scale*xv[j][k].x, scale*xv[j][k].y, scale*xv[j][k].z, scale*xv[j][k].w);
            r.x *= m[k].x;
            r.y *= m[k].y;
            r.z *= m[k].z;
            r.w *= m[k].w;

            float amax = fmaxf(fmaxf(fabsf(r.x), fabsf(r.y)), fmaxf(fabsf(r.z), fabsf(r.w)));
            amax = warp_reduce_max<8>(amax);
            const float d = amax / 127.0f;
            char4 q;
            q.x = amax == 0.0f ? 0 : roundf(r.x / d);
            q.y = amax == 0.0f ? 0 : roundf(r.y / d);
            q.z = amax == 0.0f ? 0 : roundf(r.z / d);
            q.w = amax == 0.0f ? 0 : roundf(r.w / d);
            int qsum = (int) q.x + (int) q.y + (int) q.z + (int) q.w;
#pragma unroll
            for (int offset = 4; offset > 0; offset >>= 1) {
                qsum += __shfl_xor_sync(0xffffffff, qsum, offset, 8);
            }
            block_q8_1 & yb = ys[c][i / 8];
            ((char4 *) yb.qs)[i % 8] = q;
            if (i % 8 == 0) {
                yb.ds = make_half2(d, d * qsum);
            }
        }
    }
    __syncthreads();

    float tmp[NC][FN_RPG];
    float tmp_g[NC][glu ? FN_RPG : 1];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        int u[4];
        u[0] = get_int_b4(ys[c][kbx].qs, kqs + 0);
        u[1] = get_int_b4(ys[c][kbx].qs, kqs + 0 + QI4_0);
        u[2] = get_int_b4(ys[c][kbx].qs, kqs + 1);
        u[3] = get_int_b4(ys[c][kbx].qs, kqs + 1 + QI4_0);
#pragma unroll
        for (int i = 0; i < FN_RPG; ++i) {
            tmp[c][i] = 0.0f;
            tmp[c][i] += vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v[i], u, d4[i], ys[c][kbx].ds);
            if constexpr (glu) {
                tmp_g[c][i] = 0.0f;
                tmp_g[c][i] += vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(vgt[i], u, dg4[i], ys[c][kbx].ds);
            }
        }
    }

    // the sums of the 4 warps of a group are added in order per lane, then the warp is reduced: warp gw of the group does
    // token c0 + gw with its own sums from registers, the other warps write theirs to shared memory
#pragma unroll
    for (int c0 = 0; c0 < NC; c0 += TC) {
        if (c0 > 0) {
            __syncthreads();
        }
#pragma unroll
        for (int c = c0; c < c0 + TC; ++c) {
            if (c - c0 == gw) {
                continue;
            }
#pragma unroll
            for (int i = 0; i < FN_RPG; ++i) {
                part[grp][gw][c - c0][i][lane] = tmp[c][i];
                if constexpr (glu) {
                    part_g[grp][gw][c - c0][i][lane] = tmp_g[c][i];
                }
            }
        }
        __syncthreads();
        const int c = c0 + gw;
        if (gw >= TC || c >= NC) {
            continue;
        }
#pragma unroll
        for (int i = 0; i < FN_RPG; ++i) {
            float own   = tmp[c0][i];
            float own_g = glu ? tmp_g[c0][i] : 0.0f;
#pragma unroll
            for (int k = 1; k < TC; ++k) {
                own = k == gw ? tmp[c0 + k][i] : own; // no dynamic register indexing
            }
            float sum   = gw == 0 ? own   : part[grp][0][gw][i][lane];
            float sum_g = gw == 0 ? own_g : part_g[grp][0][gw][i][lane];
#pragma unroll
            for (int l = 1; l < 4; ++l) {
                sum += l == gw ? own : part[grp][l][gw][i][lane];
                if constexpr (glu) {
                    sum_g += part_g[grp][l][gw][i][lane]; // gw == 0 (TC == 1)
                }
            }
            sum = warp_reduce_sum(sum);
            if constexpr (glu) {
                sum_g = warp_reduce_sum(sum_g);
            }
            if (lane == i && row0 + i < nrows) {
                float result = sum;
                if constexpr (glu) {
                    // as mul_mat_vec_q with the fused gate (zero biases added)
                    result += 0.0f;
                    float gate_value = sum_g;
                    gate_value += 0.0f;
                    result *= ggml_cuda_op_silu_single(gate_value);
                }
                dst[c * dst_s1 + row0 + i] = result;
            }
        }
    }
}

bool ggml_cuda_flint_norm_mmvq_supported(const ggml_tensor * x, const ggml_tensor * norm_w, const ggml_tensor * mm) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_AMPERE) {
        return false;
    }
    const ggml_tensor * w = mm->src[0];
    const int64_t n = flint_ntok(x);
    return x->type == GGML_TYPE_F32 && x->ne[0] == FR_NCOLS && n >= 1 && n <= FLINT_MAX_TOKENS &&
        ggml_is_contiguous_rows(x) && x->nb[1] % 16 == 0 && ggml_cuda_is_aligned(x, 16) &&
        norm_w->type == GGML_TYPE_F32 && norm_w->ne[0] == FR_NCOLS && ggml_nrows(norm_w) == 1 &&
        ggml_is_contiguous(norm_w) && ggml_cuda_is_aligned(norm_w, 16) &&
        w->type == GGML_TYPE_Q4_0 && w->ne[0] == FR_NCOLS && w->ne[2] == 1 && w->ne[3] == 1 &&
        w->nb[1] % sizeof(block_q4_0) == 0 && w->ne[1] < INT_MAX &&
        mm->type == GGML_TYPE_F32 && flint_ntok(mm) == n && ggml_is_contiguous(mm);
}

template <int RPG1, int RPGN, bool glu> // rows per group: one token, several tokens
static void flint_norm_mmvq_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * norm_w,
        const float eps, const ggml_tensor * wu, const ggml_tensor * wg, ggml_tensor * dst) {
    const int nrows = (int) wu->ne[1];
    const int rows = (flint_ntok(x) == 1 ? RPG1 : RPGN) * FN_WARPS / 4;
    const ggml_cuda_kernel_launch_params params(dim3((nrows + rows - 1) / rows), dim3(FN_WARPS * WARP_SIZE), 0, ctx.stream());
    const float *  xd = (const float *) x->data;
    const int64_t  xs = x->nb[1] / sizeof(float);
    const float *  nd = (const float *) norm_w->data;
    const char *   vx = (const char *) wu->data;
    const char *   vg = wg ? (const char *) wg->data : nullptr;
    const int64_t  sr = wu->nb[1] / sizeof(block_q4_0);
    float *        dd = (float *) dst->data;
    const int64_t  ds = dst->nb[1] / sizeof(float);
    static_assert(FLINT_MAX_TOKENS == 4, "one case per token count");
    switch (flint_ntok(x)) {
        case 1:  ggml_cuda_kernel_launch(flint_norm_mmvq_kernel<RPG1, glu, 1>, params, xd, xs, nd, eps, vx, vg, sr, nrows, dd, ds); break;
        case 2:  ggml_cuda_kernel_launch(flint_norm_mmvq_kernel<RPGN, glu, 2>, params, xd, xs, nd, eps, vx, vg, sr, nrows, dd, ds); break;
        case 3:  ggml_cuda_kernel_launch(flint_norm_mmvq_kernel<RPGN, glu, 3>, params, xd, xs, nd, eps, vx, vg, sr, nrows, dd, ds); break;
        default: ggml_cuda_kernel_launch(flint_norm_mmvq_kernel<RPGN, glu, 4>, params, xd, xs, nd, eps, vx, vg, sr, nrows, dd, ds); break;
    }
}

void ggml_cuda_flint_norm_mmvq(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * norm_w,
        const float eps, ggml_tensor * mm) {
    flint_norm_mmvq_launch<FN_MM_RPG, FN_MM_RPG_N, false>(ctx, x, norm_w, eps, mm->src[0], nullptr, mm);
}

bool ggml_cuda_flint_norm_glu_supported(const ggml_tensor * x, const ggml_tensor * norm_w, const ggml_tensor * up,
        const ggml_tensor * gate, const ggml_tensor * glu) {
    if (!ggml_cuda_flint_norm_mmvq_supported(x, norm_w, up) || !ggml_cuda_flint_norm_mmvq_supported(x, norm_w, gate)) {
        return false;
    }
    const ggml_tensor * wu = up->src[0];
    const ggml_tensor * wg = gate->src[0];
    return ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU && glu->src[0] == gate && glu->src[1] == up &&
        ggml_get_op_params_i32(glu, 1) == 0 && wu->ne[1] == wg->ne[1] && wu->nb[1] == wg->nb[1] &&
        glu->type == GGML_TYPE_F32 && ggml_is_contiguous(glu) && flint_ntok(glu) == flint_ntok(x) && glu->ne[0] == wu->ne[1];
}

void ggml_cuda_flint_norm_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * norm_w,
        const float eps, const ggml_tensor * up, const ggml_tensor * gate, ggml_tensor * glu) {
    flint_norm_mmvq_launch<FN_GLU_RPG, FN_GLU_RPG, true>(ctx, x, norm_w, eps, up->src[0], gate->src[0], glu);
}

// MUL_MAT (Q4_0) + ADD of NC tokens: FO_WARPS warps, each 4 warps do FO_RPG rows, every row and token as one mul_mat_vec_q
// block (4 warps, 1 row: thread t sums items t, t + 128, ... in order (item i: q4_0 block i/2, ints 2*(i%2), 2*(i%2) + 1),
// then warps 1..3 are added to warp 0 in order and the warp is reduced; the residual is added last). The weights are loaded
// before the PDL wait. The q8_1 activations of token c start at block c*y_s1.
#define FO_WARPS 8
#define FO_RPG   2
#define FO_GROUPS (FO_WARPS / 4)

template <int ipt, int NC> // items per thread, tokens
static __global__ void __launch_bounds__(FO_WARPS * WARP_SIZE) flint_mmvq_add_kernel(
        const char * vx, const int64_t stride_row, const block_q8_1 * y, const int64_t y_s1, const float * res,
        const int64_t res_s1, const int nrows, float * dst, const int64_t dst_s1) {
    constexpr int nblk = ipt * 64; // q4_0 blocks per row
    __shared__ block_q8_1 ys[NC][nblk];
    __shared__ float      part[FO_GROUPS][3][FO_RPG][WARP_SIZE];

    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int grp  = tid / 128;
    const int gt   = tid % 128;
    const int gw   = gt / WARP_SIZE;
    const int row0 = (blockIdx.x * FO_GROUPS + grp) * FO_RPG;

    ggml_cuda_pdl_lc();

    int   v[FO_RPG][ipt][2];
    float d4[FO_RPG][ipt];
#pragma unroll
    for (int r = 0; r < FO_RPG; ++r) {
#pragma unroll
        for (int j = 0; j < ipt; ++j) {
            const int item = gt + 128 * j;
            if (row0 + r < nrows) {
                const block_q4_0 * b = (const block_q4_0 *) vx + (row0 + r) * stride_row + item / 2;
                v[r][j][0] = get_int_b2(b->qs, 2 * (item % 2) + 0);
                v[r][j][1] = get_int_b2(b->qs, 2 * (item % 2) + 1);
                d4[r][j]   = b->d;
            } else {
                v[r][j][0] = v[r][j][1] = 0;
                d4[r][j]   = 0.0f;
            }
        }
    }

    ggml_cuda_pdl_sync();

    static_assert(sizeof(ys[0]) % 16 == 0, "ys is copied as int4");
    constexpr int n16 = (int) (sizeof(ys[0]) / 16);
    for (int i = tid; i < NC * n16; i += FO_WARPS * WARP_SIZE) {
        ((int4 *) ys[i / n16])[i % n16] = ((const int4 *) (y + (i / n16) * y_s1))[i % n16];
    }
    float rv[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        rv[c] = gw == 0 && lane < FO_RPG && row0 + lane < nrows ? res[c * res_s1 + row0 + lane] : 0.0f;
    }
    __syncthreads();

    float tmp[NC][FO_RPG];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int r = 0; r < FO_RPG; ++r) {
            tmp[c][r] = 0.0f;
        }
    }
#pragma unroll
    for (int j = 0; j < ipt; ++j) {
        const int item = gt + 128 * j;
        const int kqs = 2 * (item % 2);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const block_q8_1 & yb = ys[c][item / 2];
            int u[4];
            u[0] = get_int_b4(yb.qs, kqs + 0);
            u[1] = get_int_b4(yb.qs, kqs + 0 + QI4_0);
            u[2] = get_int_b4(yb.qs, kqs + 1);
            u[3] = get_int_b4(yb.qs, kqs + 1 + QI4_0);
#pragma unroll
            for (int r = 0; r < FO_RPG; ++r) {
                tmp[c][r] += vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v[r][j], u, d4[r][j], yb.ds);
            }
        }
    }

#pragma unroll
    for (int c = 0; c < NC; ++c) {
        if (c > 0) {
            __syncthreads();
        }
        if (gw > 0) {
#pragma unroll
            for (int r = 0; r < FO_RPG; ++r) {
                part[grp][gw - 1][r][lane] = tmp[c][r];
            }
        }
        __syncthreads();
        if (gw > 0) {
            continue;
        }
#pragma unroll
        for (int r = 0; r < FO_RPG; ++r) {
#pragma unroll
            for (int l = 0; l < 3; ++l) {
                tmp[c][r] += part[grp][l][r][lane];
            }
            tmp[c][r] = warp_reduce_sum(tmp[c][r]);
            if (lane == r && row0 + r < nrows) {
                dst[c * dst_s1 + row0 + r] = tmp[c][r] + rv[c];
            }
        }
    }
}

bool ggml_cuda_flint_mmvq_add_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, const ggml_tensor * add) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_AMPERE) {
        return false;
    }
    const ggml_tensor * w   = mm->src[0];
    const ggml_tensor * x   = mm->src[1];
    const ggml_tensor * res = add->src[0] == mm ? add->src[1] : add->src[0];
    const int64_t ipt = w->ne[0] / (QK4_0 * 64);
    const int64_t n   = flint_ntok(x);
    return w->type == GGML_TYPE_Q4_0 && w->ne[0] % (QK4_0 * 64) == 0 && ipt >= 1 && ipt <= 4 && w->ne[2] == 1 &&
        w->ne[3] == 1 && w->ne[1] < INT_MAX && w->nb[1] % sizeof(block_q4_0) == 0 &&
        n >= 1 && n <= FLINT_MAX_TOKENS &&
        x->type == GGML_TYPE_F32 && ggml_is_contiguous(x) &&
        (ctx.q8_1_lookup(x) != nullptr || (x->ne[0] % MATRIX_ROW_PADDING == 0 && x->ne[0] <= ggml_backend_cuda_context::q8_1_max_ncols)) &&
        mm->type == GGML_TYPE_F32 && flint_ntok(mm) == n &&
        res->type == GGML_TYPE_F32 && ggml_are_same_shape(res, mm) && ggml_is_contiguous(res) &&
        add->type == GGML_TYPE_F32 && ggml_are_same_shape(add, mm) && ggml_is_contiguous(add);
}

template <int ipt>
static void flint_mmvq_add_launch(const ggml_cuda_kernel_launch_params & params, const int n, const char * vx, const int64_t sr,
        const block_q8_1 * y, const int64_t ys, const float * res, const int64_t rs, const int nrows, float * dst, const int64_t ds) {
    static_assert(FLINT_MAX_TOKENS == 4, "one case per token count");
    switch (n) {
        case 1:  ggml_cuda_kernel_launch(flint_mmvq_add_kernel<ipt, 1>, params, vx, sr, y, ys, res, rs, nrows, dst, ds); break;
        case 2:  ggml_cuda_kernel_launch(flint_mmvq_add_kernel<ipt, 2>, params, vx, sr, y, ys, res, rs, nrows, dst, ds); break;
        case 3:  ggml_cuda_kernel_launch(flint_mmvq_add_kernel<ipt, 3>, params, vx, sr, y, ys, res, rs, nrows, dst, ds); break;
        default: ggml_cuda_kernel_launch(flint_mmvq_add_kernel<ipt, 4>, params, vx, sr, y, ys, res, rs, nrows, dst, ds); break;
    }
}

void ggml_cuda_flint_mmvq_add(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, ggml_tensor * add) {
    const ggml_tensor * w   = mm->src[0];
    const ggml_tensor * res = add->src[0] == mm ? add->src[1] : add->src[0];
    const int nrows = (int) w->ne[1];
    const int ipt   = (int) (w->ne[0] / (QK4_0 * 64));
    const ggml_tensor * x = mm->src[1];
    const int n = (int) flint_ntok(x);
    if (ctx.q8_1_lookup(x) == nullptr) {
        // as mul_mat_vec_q quantizes it (the weights of this kernel are loaded during the quantization)
        void * q = ctx.q8_1_next_buf();
        quantize_row_q8_1_cuda((const float *) x->data, q, x->ne[0], x->ne[0], x->ne[0]*n, x->ne[0]*n, x->ne[0], n, 1, 1, ctx.stream());
        ctx.q8_1_set(x);
    }
    const block_q8_1 * y = (const block_q8_1 *) ctx.q8_1_lookup(x);
    const int64_t ys = x->ne[0] / QK8_1;
    const ggml_cuda_kernel_launch_params params(dim3((nrows + FO_GROUPS * FO_RPG - 1) / (FO_GROUPS * FO_RPG)),
        dim3(FO_WARPS * WARP_SIZE), 0, ctx.stream());
    const char *  vx = (const char *) w->data;
    const int64_t sr = w->nb[1] / sizeof(block_q4_0);
    const float * rd = (const float *) res->data;
    const int64_t rs = res->nb[1] / sizeof(float);
    float *       dd = (float *) add->data;
    const int64_t ds = add->nb[1] / sizeof(float);
    switch (ipt) {
        case 1:  flint_mmvq_add_launch<1>(params, n, vx, sr, y, ys, rd, rs, nrows, dd, ds); break;
        case 2:  flint_mmvq_add_launch<2>(params, n, vx, sr, y, ys, rd, rs, nrows, dd, ds); break;
        case 3:  flint_mmvq_add_launch<3>(params, n, vx, sr, y, ys, rd, rs, nrows, dd, ds); break;
        default: flint_mmvq_add_launch<4>(params, n, vx, sr, y, ys, rd, rs, nrows, dd, ds); break;
    }
}

bool ggml_cuda_flint_route_supported(ggml_backend_cuda_context & ctx, const ggml_cuda_flint_route_args & a) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_AMPERE) {
        return false;
    }
    const int64_t n = flint_ntok(a.x);
    return a.x->type == GGML_TYPE_F32 && a.x->ne[0] == FR_NCOLS && n >= 1 && n <= FR_MAX_TOKENS && ggml_is_contiguous_rows(a.x) &&
        a.x->nb[1] % 16 == 0 &&
        a.norm_w->type == GGML_TYPE_F32 && a.norm_w->ne[0] == FR_NCOLS && ggml_nrows(a.norm_w) == 1 &&
        a.xn->type == GGML_TYPE_F32 && ggml_is_contiguous(a.xn) && flint_ntok(a.xn) == n &&
        a.router->type == GGML_TYPE_BF16 && a.router->ne[0] == FR_NCOLS && a.router->ne[1] == FR_NEXPERT &&
        a.router->nb[1] % 16 == 0 &&
        a.bias != nullptr && a.bias->type == GGML_TYPE_F32 && a.bias->ne[0] == FR_NEXPERT && ggml_is_contiguous(a.bias) &&
        a.ids->type == GGML_TYPE_I32 && a.ids->ne[0] <= FLINT_MAX_USED && flint_ntok(a.ids) == n &&
        a.weights->type == GGML_TYPE_F32 && ggml_nelements(a.weights) == a.ids->ne[0] * n && ggml_is_contiguous(a.weights) &&
        ggml_cuda_is_aligned(a.x, 16) && ggml_cuda_is_aligned(a.norm_w, 16) && ggml_cuda_is_aligned(a.xn, 16) &&
        ggml_cuda_is_aligned(a.router, 16) && ctx.fused_scratch_ready();
}

void ggml_cuda_flint_route(ggml_backend_cuda_context & ctx, const ggml_cuda_flint_route_args & a) {
    static_assert(FR_NCOLS <= ggml_backend_cuda_context::q8_1_max_ncols, "q8_1 buffer too small");
    block_q8_1 * yq = (block_q8_1 *) ctx.q8_1_next_buf();
    static_assert(sizeof(flint_route_scratch) <= ggml_backend_cuda_context::fused_scratch_mmid_offset, "scratch too small");
    flint_route_scratch * scratch = (flint_route_scratch *) ctx.get_fused_scratch();
    const int n = (int) flint_ntok(a.x);
    const size_t smem = n * FR_NCOLS * sizeof(float); // the normalized activations
    const ggml_cuda_kernel_launch_params params(dim3(FR_BLOCKS, 1, 1), dim3(FR_THREADS, 1, 1), smem, ctx.stream());
    const int n_used = (int) a.ids->ne[0];
    const auto launch = [&](auto kernel) {
        static bool smem_set[GGML_CUDA_MAX_DEVICES][FR_MAX_TOKENS] = {}; // per kernel (token count)
        if (!smem_set[ctx.device][n - 1]) {
            CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
            smem_set[ctx.device][n - 1] = true;
        }
        ggml_cuda_kernel_launch(kernel, params,
            (const float *) a.x->data, (int64_t) (a.x->nb[1] / sizeof(float)), (const float *) a.norm_w->data,
            (float *) a.xn->data, (int64_t) (a.xn->nb[1] / sizeof(float)), yq,
            (const char *) a.router->data, (int64_t) a.router->nb[1], (const float *) a.bias->data, scratch,
            (int32_t *) a.ids->data, (int64_t) (a.ids->nb[1] / sizeof(int32_t)), (float *) a.weights->data, (int64_t) n_used,
            n_used, a.eps, a.clamp, a.scale, a.norm, a.defer);
    };
    static_assert(FR_MAX_TOKENS == 8, "one case per token count");
    switch (n) {
        case 1:  launch(flint_route_kernel<1>); break;
        case 2:  launch(flint_route_kernel<2>); break;
        case 3:  launch(flint_route_kernel<3>); break;
        case 4:  launch(flint_route_kernel<4>); break;
        case 5:  launch(flint_route_kernel<5>); break;
        case 6:  launch(flint_route_kernel<6>); break;
        case 7:  launch(flint_route_kernel<7>); break;
        default: launch(flint_route_kernel<8>); break;
    }
    ctx.q8_1_set(a.xn);
    if (a.defer) {
        ggml_cuda_topk_deferred & d = ctx.topk_deferred;
        d.ids_src = a.ids->data;
        d.logits  = scratch->logits[0];
        d.bias    = (const float *) a.bias->data;
        d.ids      = (int32_t *) a.ids->data;
        d.weights  = (float *) a.weights->data;
        d.ids_s1   = a.ids->nb[1] / sizeof(int32_t);
        d.w_s1     = n_used;
        d.n_tokens = (int) flint_ntok(a.x);
        d.n_used   = n_used;
        d.norm    = a.norm;
        d.clamp   = a.clamp;
        d.scale   = a.scale;
    }
}

// block t: the top-k of token t
static __global__ void __launch_bounds__(FLINT_NEXPERT) flint_topk_kernel(const ggml_cuda_topk_deferred d) {
    const int   t      = blockIdx.x;
    const float bias_v = flint_topk_bias(d.bias);
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    __shared__ int   ids_s[FLINT_MAX_USED];
    __shared__ float w_s[FLINT_MAX_USED];
    flint_topk_block(d.logits + t * FLINT_NEXPERT, bias_v, d.n_used, d.norm, d.clamp, d.scale, ids_s, w_s);
    if ((int) threadIdx.x < d.n_used) {
        d.ids[t * d.ids_s1 + threadIdx.x]   = ids_s[threadIdx.x];
        d.weights[t * d.w_s1 + threadIdx.x] = w_s[threadIdx.x];
    }
}

void ggml_cuda_flint_topk_flush(ggml_backend_cuda_context & ctx) {
    if (ctx.topk_deferred.ids_src == nullptr) {
        return;
    }
    const ggml_cuda_kernel_launch_params params(dim3(ctx.topk_deferred.n_tokens, 1, 1), dim3(FLINT_NEXPERT, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(flint_topk_kernel, params, ctx.topk_deferred);
    ctx.topk_deferred = {};
}

// Attention: grid (FA_GQA/hpb*n_split, n_head_kv, n_tokens), block of FA_WARPS warps: block (x, g) handles hpb Q heads of KV
// head g (with splits all FA_GQA, else FA_HPB_1: more blocks for the latency bound work outside the chunks) and the splits
// x/(FA_GQA/hpb) + i*n_blocks_per_head. The chunks of FA_CHUNK cells with unmasked cells are spread evenly over the splits
// (the split count depends only on the mask of the token). K and V of the chunks of a split are loaded into shared memory,
// the next chunk while the current one is processed (the first two before the PDL wait: the cache rows of earlier tokens are
// already written, the mask and the cell indices are graph inputs). Warp w takes the cells [16*w, 16*w + 16) of each chunk:
// S^T = K*Q^T and O^T += V^T*P^T with mma (f16 inputs, f32 accumulators; cells or dims are the 16 rows, the heads the 8
// columns), with a softmax state per warp. The states of the warps, then of the splits (by the first blocks of a KV head, a
// head each, once all splits are written), are combined in a fixed order. The K/V of the current token are computed locally
// and replace that cell in shared memory, block 0 also writes them to the cache. The blocks of a KV head read its Q rows
// before any output is written, so dst may reuse the memory of Q (see ggml_cuda_flint_attn_supported).
#define FA_D      128
#define FA_GQA    8  // Q heads per KV head: the 8 columns of the mma
#define FA_CHUNK  64
#define FA_WARPS  (FA_CHUNK/16)
#define FA_STAGES 2  // chunks in shared memory: the next chunk loads while the current one is processed
#define FA_KSTR   (FA_D + 8) // K/V/Q rows of 272 bytes: the 8 rows of an ldmatrix hit different banks
#define FA_MAX_KV_HEADS 8
#define FA_MAX_CPS 64    // max. chunks of a split
#define FA_MAX_KV  12288 // the mask of all cells is kept in shared memory; longer: stock kernel
#define FA_CH_WORDS ((FA_MAX_KV/FA_CHUNK + 63)/64) // words of the chunk bit mask
#define FA_MIN_SPLIT_CHUNKS 5 // fewer chunks are not split: combining the splits costs as much as a few chunks
#define FA_HPB_1   2 // Q heads per block without splits

// partial softmax states of the splits: per head O, max, sum (padded to 16 bytes); per (token, KV head) two counters: the
// splits written and the combining blocks done waiting
#define FA_HP   (FA_D + 4)
#define FA_PART (FA_GQA*FA_HP)
static constexpr size_t fa_counters_offset = (sizeof(flint_route_scratch) + 255) / 256 * 256;
static_assert(fa_counters_offset + 2*FLINT_ATTN_MAX_TOKENS*FA_MAX_KV_HEADS*sizeof(unsigned int) <= ggml_backend_cuda_context::fused_scratch_mmid_offset,
    "scratch too small");

struct flint_attn_rope {
    float          freq_scale;
    float          ext_factor;
    float          attn_factor;
    float          theta_scale;
    float          corr_dims[2];
};

static __device__ __forceinline__ float flint_rope_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// same as rope_yarn<true> in rope.cu
static __device__ __forceinline__ void flint_rope_yarn(const float theta_extrap, const flint_attn_rope & rp, const int i0,
        float & cos_theta, float & sin_theta) {
    const float theta_interp = rp.freq_scale * theta_extrap;
    float theta  = theta_interp;
    float mscale = rp.attn_factor;
    if (rp.ext_factor != 0.0f) {
        const float ramp_mix = flint_rope_ramp(rp.corr_dims[0], rp.corr_dims[1], i0) * rp.ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / rp.freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
}

// cos/sin for the dims of the lane (see flint_norm_rope), as rms_norm_mul_rope_f32 computes them
template <int rope_mode>
static __device__ __forceinline__ void flint_rope_cs(const int pos, const flint_attn_rope & rp, float * c, float * s) {
    const int lane = threadIdx.x % WARP_SIZE;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int i0 = rope_mode == 1 ? 4*lane + (k & 2) : 2*(4*(lane % 16) + k);
        if (rope_mode != 0 && (rope_mode == 2 || (k & 1) == 0)) {
            flint_rope_yarn(pos*powf(rp.theta_scale, i0/2.0f), rp, i0, c[k], s[k]);
        } else {
            c[k] = 1.0f;
            s[k] = 0.0f;
        }
    }
}

// RMS norm * weight (+ rope) of one row of FA_D, lane holds dims 4*lane .. 4*lane + 3
template <int rope_mode>
static __device__ __forceinline__ float4 flint_norm_rope(float4 x, const float4 w, const float eps, const float * c, const float * s) {
    const int lane = threadIdx.x % WARP_SIZE;
    float ss = x.x*x.x + x.y*x.y + x.z*x.z + x.w*x.w;
    ss = warp_reduce_sum(ss);
    const float scale = rsqrtf(ss/FA_D + eps);
    x.x = scale*x.x*w.x;
    x.y = scale*x.y*w.y;
    x.z = scale*x.z*w.z;
    x.w = scale*x.w*w.w;
    if constexpr (rope_mode == 1) {
        // normal: pairs (2i, 2i + 1)
        const float x0 = x.x, x1 = x.y, x2 = x.z, x3 = x.w;
        x.x = x0*c[0] - x1*s[0];
        x.y = x0*s[0] + x1*c[0];
        x.z = x2*c[2] - x3*s[2];
        x.w = x2*s[2] + x3*c[2];
    } else if constexpr (rope_mode == 2) {
        // neox: pairs (i, i + FA_D/2), lane l pairs with lane l ^ 16
        const float4 y = make_float4(__shfl_xor_sync(0xFFFFFFFF, x.x, 16), __shfl_xor_sync(0xFFFFFFFF, x.y, 16),
                                     __shfl_xor_sync(0xFFFFFFFF, x.z, 16), __shfl_xor_sync(0xFFFFFFFF, x.w, 16));
        const bool hi = lane >= 16;
        const float xv[4] = { x.x, x.y, x.z, x.w };
        const float yv[4] = { y.x, y.y, y.z, y.w };
        float r[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float x0 = hi ? yv[k] : xv[k];
            const float x1 = hi ? xv[k] : yv[k];
            r[k] = hi ? x0*s[k] + x1*c[k] : x0*c[k] - x1*s[k];
        }
        x = make_float4(r[0], r[1], r[2], r[3]);
    }
    return x;
}

static __device__ __forceinline__ uint2 flint_f4_to_h4(const float4 v) {
    const half2 a = __floats2half2_rn(v.x, v.y);
    const half2 b = __floats2half2_rn(v.z, v.w);
    return make_uint2(*(const uint32_t *) &a, *(const uint32_t *) &b);
}

static __device__ __forceinline__ void flint_ldsm_x4(uint32_t (&r)[4], const half * p) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((unsigned int) __cvta_generic_to_shared(p)));
#else
    GGML_UNUSED_VARS(r, p);
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ void flint_ldsm_x4_trans(uint32_t (&r)[4], const half * p) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((unsigned int) __cvta_generic_to_shared(p)));
#else
    GGML_UNUSED_VARS(r, p);
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ void flint_ldsm_x2_trans(uint32_t (&r)[2], const half * p) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
        : "=r"(r[0]), "=r"(r[1]) : "r"((unsigned int) __cvta_generic_to_shared(p)));
#else
    GGML_UNUSED_VARS(r, p);
    NO_DEVICE_CODE;
#endif
}

// c += a*b, m16n8k16 (a: 16x16 row-major, b: 16x8 column-major), f16 inputs, f32 accumulators
static __device__ __forceinline__ void flint_mma(float (&c)[4], const uint32_t (&a)[4], const uint32_t b0, const uint32_t b1) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(c, a, b0, b1);
    NO_DEVICE_CODE;
#endif
}

// 16 bytes to shared memory, zeros if !valid
static __device__ __forceinline__ void flint_cp_async_16(half * dst, const half * src, const bool valid) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
        :: "r"((unsigned int) __cvta_generic_to_shared(dst)), "l"(src), "r"(valid ? 16 : 0));
#else
    GGML_UNUSED_VARS(dst, src, valid);
    NO_DEVICE_CODE;
#endif
}

// Several tokens (blockIdx.z): token t is row t % ne1 of stream t / ne1 of the K/V views (stream i starts at cache row
// kv_row0 + i*kv_rows). Every block computes K/V of all tokens of its stream and puts them in place of the cells they write,
// so tokens of a batch see each other without reading cache rows written by other blocks.
template <int rope_mode>
static __global__ void __launch_bounds__(FA_WARPS*WARP_SIZE, 1) flint_attn_kernel(
        const float * q, const int64_t q_s1, const int64_t q_s2, const float * k, const int64_t k_s1, const int64_t k_s2,
        const float * v, const int64_t v_s1, const int64_t v_s2,
        const float * qw, const float * kw, const float eps_q, const float eps_k,
        const int32_t * pos_p, const flint_attn_rope rp,
        half * kc, half * vc, const int64_t kc_s1, const int64_t vc_s1, const int64_t kc_s3, const int64_t vc_s3,
        const int64_t * kidx, const int64_t * vidx, const int64_t k_row0, const int64_t v_row0, const int64_t kv_rows,
        const half * mask, const int64_t mask_s1, const int64_t mask_s3, const int ne1, const int n_kv, const int nsp_max, const int hpb,
        const float * sinks, const float kq_scale, float * dst, const bool f32_out, block_q8_1 * yq,
        float * part, unsigned int * counters) {
    constexpr int nthreads = FA_WARPS*WARP_SIZE;
    constexpr int OSTR     = FA_D + 4; // rows of the warp states: the 4 heads of a store hit different banks

    extern __shared__ __align__(16) char fa_smem[];
    half  (*Ks)[FA_CHUNK][FA_KSTR] = (half (*)[FA_CHUNK][FA_KSTR]) fa_smem;
    half  (*Vs)[FA_CHUNK][FA_KSTR] = (half (*)[FA_CHUNK][FA_KSTR]) (fa_smem + FA_STAGES*FA_CHUNK*FA_KSTR*sizeof(half));
    half  * mask_s                 = (half *) (fa_smem + 2*FA_STAGES*FA_CHUNK*FA_KSTR*sizeof(half));
    float (*Os)[FA_GQA][OSTR]      = (float (*)[FA_GQA][OSTR]) fa_smem; // O of the warps (the K buffers are free then)
    static_assert(FA_WARPS*FA_GQA*OSTR*sizeof(float) <= FA_STAGES*FA_CHUNK*FA_KSTR*sizeof(half), "smem too small");

    __shared__ __align__(16) half qh[FA_GQA][FA_KSTR];
    __shared__ __align__(16) half ps[FA_WARPS][16][FA_GQA]; // P of the warp: cells x heads
    __shared__ float Ms[FA_WARPS][FA_GQA];
    __shared__ float Ls[FA_WARPS][FA_GQA];
    __shared__ __align__(16) half kcur[FLINT_MAX_TOKENS][FA_D]; // the tokens of the stream
    __shared__ __align__(16) half vcur[FLINT_MAX_TOKENS][FA_D];
    __shared__ int kcell[FLINT_MAX_TOKENS]; // cell of the stream that token j of the stream writes (or -1)
    __shared__ int vcell[FLINT_MAX_TOKENS];
    __shared__ float sinks_s[FA_GQA];
    __shared__ unsigned long long valid_ch[FA_CH_WORDS]; // bit per chunk with unmasked cells
    __shared__ int chunk_id[FA_MAX_CPS];                 // chunks of the split

    const int tid    = threadIdx.x;
    const int lane   = tid % WARP_SIZE;
    const int w      = tid / WARP_SIZE;
    const int g      = blockIdx.y;
    const int n_head = FA_GQA*gridDim.y;
    const int nhb    = FA_GQA/hpb;           // blocks of heads per KV head
    const int hb     = blockIdx.x % nhb;
    const int nbs    = gridDim.x / nhb;      // blocks per head: block i processes the splits i, i + nbs, ...
    const int h0     = g*FA_GQA + hb*hpb;    // the first head of the block, the columns of the others are not used
    const int t      = blockIdx.z;
    const int st     = t / ne1;    // stream
    const int t0     = st*ne1;     // first token of the stream
    const int nst    = ne1;        // tokens of the stream

    q    += t*q_s2;
    k    += t0*k_s2;
    v    += t0*v_s2;
    kc   += st*kc_s3;
    vc   += st*vc_s3;
    mask += st*mask_s3 + (t - t0)*mask_s1;
    dst  += (int64_t) t*n_head*FA_D;
    yq   += (int64_t) t*n_head*FA_D/QK8_1;

    // graph inputs, weights and cache rows of earlier tokens: safe to read before the wait. All loads first.
    constexpr int mask_iters = (FA_MAX_KV + 8*nthreads - 1)/(8*nthreads);
    static_assert((8*WARP_SIZE) % FA_CHUNK == 0 && 64 % (8*WARP_SIZE/FA_CHUNK) == 0, "a warp covers whole chunks of one word");
    const int64_t srow_k = k_row0 + st*kv_rows; // cache row of cell 0 of the stream
    const int64_t srow_v = v_row0 + st*kv_rows;
    if (tid < nst) {
        const int64_t ck = kidx[t0 + tid] - srow_k;
        const int64_t cv = vidx[t0 + tid] - srow_v;
        kcell[tid] = ck >= 0 && ck < kv_rows ? (int) ck : -1;
        vcell[tid] = cv >= 0 && cv < kv_rows ? (int) cv : -1;
        static_assert(FLINT_MAX_TOKENS <= nthreads, "one thread per token");
    }
    const int     pos = rope_mode != 0 ? pos_p[t] : 0;
    const float4  qwv = ((const float4 *) qw)[lane];
    const float4  kwv = ((const float4 *) kw)[lane];
    if (tid < FA_GQA) {
        sinks_s[tid] = sinks && tid < hpb ? sinks[h0 + tid] : -INFINITY;
    }
    float rc[4];
    float rs[4];
    flint_rope_cs<rope_mode>(pos, rp, rc, rs);

    // the mask of the token goes to shared memory, the chunks with unmasked cells are counted (a warp covers whole chunks)
    {
        uint4 m8[mask_iters];
#pragma unroll
        for (int it = 0; it < mask_iters; ++it) {
            const int c = 8*(tid + it*nthreads); // n_kv % 8 == 0 is checked on the host
            m8[it] = c < n_kv ? *(const uint4 *) (mask + c) : make_uint4(0xFC00FC00, 0xFC00FC00, 0xFC00FC00, 0xFC00FC00);
        }
        if (tid < FA_CH_WORDS) {
            valid_ch[tid] = 0;
        }
        __syncthreads();
#pragma unroll
        for (int it = 0; it < mask_iters; ++it) {
            const int c = 8*(tid + it*nthreads);
            if (c < n_kv) {
                *(uint4 *) (mask_s + c) = m8[it];
            }
            const half * mh = (const half *) &m8[it];
            bool any = false;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                any = any || __half2float(mh[i]) != -INFINITY;
            }
            const unsigned int bal = __ballot_sync(0xFFFFFFFF, any);
            const int c_w = c - 8*lane; // first cell of the warp
            if (lane == 0 && bal) {
                unsigned long long bits = 0;
#pragma unroll
                for (int i = 0; i < 8*WARP_SIZE/FA_CHUNK; ++i) {
                    bits |= (unsigned long long) ((bal >> (i*FA_CHUNK/8)) & ((1u << (FA_CHUNK/8)) - 1) ? 1 : 0) << i;
                }
                atomicOr(&valid_ch[c_w/FA_CHUNK/64], bits << (c_w/FA_CHUNK % 64));
            }
        }
        __syncthreads();
    }
    int n_valid = 0;
#pragma unroll
    for (int i = 0; i < FA_CH_WORDS; ++i) {
        n_valid += __popcll(valid_ch[i]);
    }
    // the next kernel may start (it cannot run before this one completes): without splits a few blocks finish quickly and
    // the next kernel loads its weights now, with splits only after the chunk loads (it would slow down the many blocks)
    if (nsp_max == 1) {
        ggml_cuda_pdl_lc();
    }
    // splits used by the token
    const int nsp = n_valid < FA_MIN_SPLIT_CHUNKS ? 1 : min(nsp_max, n_valid);

    // mma accumulators: lane holds the heads hq and hq + 1 of the rows (cells or dims) r and r + 8
    const int hq = (lane % 4)*2;
    const int r  = lane / 4;
    uint32_t qb[FA_D/16][2]; // Q^T: the B fragments of S^T = K*Q^T

    // combined state: 16 threads per head, 8 dims each
    const int hh = tid / 16;
    const int d0 = (tid % 16)*8;
    static_assert(FA_GQA*16 == nthreads, "16 threads per head");
    float Mt = -INFINITY;
    float Lt = 0.0f;
    float o[8];

    int n_mine = 0; // splits processed by the block
    for (int sp = blockIdx.x / nhb; sp < nsp; sp += nbs, ++n_mine) {
        // the chunks with unmasked cells are spread evenly over the splits: split sp gets those of rank [lo, lo + n_loc)
        const int lo    = sp*n_valid/nsp;
        const int n_loc = (sp + 1)*n_valid/nsp - lo;
        if (n_mine > 0) {
            __syncthreads(); // the previous split is done with the chunks
        }
        for (int c = tid; c < FA_CH_WORDS*64; c += nthreads) {
            const unsigned long long bits = valid_ch[c / 64];
            const int b = c % 64;
            if ((bits >> b) & 1) {
                int rank = __popcll(bits & ((1ull << b) - 1)) - lo;
                for (int i = 0; i < c / 64; ++i) {
                    rank += __popcll(valid_ch[i]);
                }
                if (rank >= 0 && rank < n_loc) {
                    chunk_id[rank] = c;
                }
            }
        }
        __syncthreads();

        // chunk i of the split to buffer b, the rows of masked cells are zeroed (p is 0, V must not be NaN)
        const auto load_chunk = [&](const int i, const int b) {
            const int ch = chunk_id[i];
#pragma unroll
            for (int it = 0; it < 2*FA_CHUNK*(FA_D/8)/nthreads; ++it) {
                const int  id   = tid + it*nthreads;
                const int  is_v = id / (FA_CHUNK*(FA_D/8));
                const int  j    = id / (FA_D/8) % FA_CHUNK;
                const int  part = id % (FA_D/8);
                const int  c    = ch*FA_CHUNK + j;
                const bool ok   = c < n_kv && __half2float(mask_s[c]) != -INFINITY;
                const int64_t cr = ok ? c : 0;
                if (is_v) {
                    flint_cp_async_16(&Vs[b][j][8*part], vc + cr*vc_s1 + g*FA_D + 8*part, ok);
                } else {
                    flint_cp_async_16(&Ks[b][j][8*part], kc + cr*kc_s1 + g*FA_D + 8*part, ok);
                }
            }
            cp_async_commit_group();
        };

        if (n_loc > 0) {
            load_chunk(0, 0);
        }
        if (n_loc > 1) {
            load_chunk(1, 1);
        }

        if (n_mine == 0) {
            if (nsp_max > 1) {
                ggml_cuda_pdl_lc();
            }
            ggml_cuda_pdl_sync();

            // rows: the Q heads of the block, K and V of the stream tokens; the columns of the other heads are zero
            for (int i = tid; i < (FA_GQA - hpb)*(FA_D/4); i += nthreads) {
                ((uint2 *) qh[hpb + i/(FA_D/4)])[i % (FA_D/4)] = make_uint2(0, 0);
            }
            for (int rr = w; rr < hpb + 2*nst; rr += FA_WARPS) {
                if (rr < hpb) {
                    const float4 x = flint_norm_rope<rope_mode>(((const float4 *) (q + (h0 + rr)*q_s1))[lane], qwv, eps_q, rc, rs);
                    ((uint2 *) qh[rr])[lane] = flint_f4_to_h4(make_float4(x.x*kq_scale, x.y*kq_scale, x.z*kq_scale, x.w*kq_scale));
                } else if (rr < hpb + nst) {
                    const int j = rr - hpb;
                    float jc[4];
                    float js[4];
                    if (t0 + j == t) {
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            jc[i] = rc[i];
                            js[i] = rs[i];
                        }
                    } else {
                        flint_rope_cs<rope_mode>(rope_mode != 0 ? pos_p[t0 + j] : 0, rp, jc, js);
                    }
                    const uint2 kh = flint_f4_to_h4(flint_norm_rope<rope_mode>(((const float4 *) (k + j*k_s2 + g*k_s1))[lane], kwv, eps_k, jc, js));
                    ((uint2 *) kcur[j])[lane] = kh;
                    if (blockIdx.x == 0 && t0 + j == t && kcell[j] >= 0) {
                        *(uint2 *) (kc + (int64_t) kcell[j]*kc_s1 + g*FA_D + 4*lane) = kh;
                    }
                } else {
                    const int j = rr - hpb - nst;
                    const uint2 vh = flint_f4_to_h4(((const float4 *) (v + j*v_s2 + g*v_s1))[lane]);
                    ((uint2 *) vcur[j])[lane] = vh;
                    if (blockIdx.x == 0 && t0 + j == t && vcell[j] >= 0) {
                        *(uint2 *) (vc + (int64_t) vcell[j]*vc_s1 + g*FA_D + 4*lane) = vh;
                    }
                }
            }
            __syncthreads();
#pragma unroll
            for (int kk = 0; kk < FA_D/16; ++kk) {
                qb[kk][0] = *(const uint32_t *) &qh[r][16*kk + hq];
                qb[kk][1] = *(const uint32_t *) &qh[r][16*kk + hq + 8];
            }
        }

        float M[2] = { -INFINITY, -INFINITY };
        float L[2] = { 0.0f, 0.0f };
        float O[FA_D/16][4] = {};

        static_assert(FA_STAGES == 2, "double buffer");
        for (int i = 0, b = 0; i < n_loc; ++i, b ^= 1) {
            if (i + 1 < n_loc) {
                cp_async_wait_group<1>();
            } else {
                cp_async_wait_group<0>();
            }
            __syncthreads();
            const int c0 = chunk_id[i]*FA_CHUNK;
            static_assert(2*FLINT_MAX_TOKENS*(FA_D/8) <= nthreads, "one thread per 16 bytes of the batch K/V");
            if (tid < 2*nst*(FA_D/8)) {
                const int is_v = tid / (nst*(FA_D/8));
                const int j    = tid / (FA_D/8) % nst;
                const int part = tid % (FA_D/8);
                const int cc   = is_v ? vcell[j] : kcell[j]; // -1: none
                if (cc >= c0 && cc < c0 + FA_CHUNK) {
                    if (is_v) {
                        ((uint4 *) Vs[b][cc - c0])[part] = ((const uint4 *) vcur[j])[part];
                    } else {
                        ((uint4 *) Ks[b][cc - c0])[part] = ((const uint4 *) kcur[j])[part];
                    }
                }
            }
            __syncthreads();

            // S^T of the 16 cells of the warp (two accumulators: shorter dependency chains)
            const int cw = c0 + 16*w;
            float sa[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            float sb[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll
            for (int kk = 0; kk < FA_D/16; kk += 2) {
                uint32_t a[4];
                flint_ldsm_x4(a, &Ks[b][16*w + lane % 16][16*kk + (lane / 16)*8]);
                flint_mma(sa, a, qb[kk][0], qb[kk][1]);
                flint_ldsm_x4(a, &Ks[b][16*w + lane % 16][16*(kk + 1) + (lane / 16)*8]);
                flint_mma(sb, a, qb[kk + 1][0], qb[kk + 1][1]);
            }
            const float m0 = cw + r     < n_kv ? __half2float(mask_s[cw + r])     : -INFINITY;
            const float m1 = cw + r + 8 < n_kv ? __half2float(mask_s[cw + r + 8]) : -INFINITY;
            float s[4];
            s[0] = m0 != -INFINITY ? (sa[0] + sb[0]) + m0 : -INFINITY;
            s[1] = m0 != -INFINITY ? (sa[1] + sb[1]) + m0 : -INFINITY;
            s[2] = m1 != -INFINITY ? (sa[2] + sb[2]) + m1 : -INFINITY;
            s[3] = m1 != -INFINITY ? (sa[3] + sb[3]) + m1 : -INFINITY;
            float mt0 = fmaxf(s[0], s[2]);
            float mt1 = fmaxf(s[1], s[3]);
#pragma unroll
            for (int off = 4; off < WARP_SIZE; off *= 2) {
                mt0 = fmaxf(mt0, __shfl_xor_sync(0xFFFFFFFF, mt0, off));
                mt1 = fmaxf(mt1, __shfl_xor_sync(0xFFFFFFFF, mt1, off));
            }
            // the mask is the same for all heads: either all heads of the warp have unmasked cells or none
            if (mt0 != -INFINITY) {
                const float Mn0 = fmaxf(M[0], mt0);
                const float Mn1 = fmaxf(M[1], mt1);
                const float f0  = expf(M[0] - Mn0);
                const float f1  = expf(M[1] - Mn1);
                const float p0  = expf(s[0] - Mn0);
                const float p1  = expf(s[1] - Mn1);
                const float p2  = expf(s[2] - Mn0);
                const float p3  = expf(s[3] - Mn1);
                // per lane: the cells of the lane (summed over the lanes of a head at the end)
                L[0] = L[0]*f0 + (p0 + p2);
                L[1] = L[1]*f1 + (p1 + p3);
                M[0] = Mn0;
                M[1] = Mn1;
#pragma unroll
                for (int mt = 0; mt < FA_D/16; ++mt) {
                    O[mt][0] *= f0;
                    O[mt][1] *= f1;
                    O[mt][2] *= f0;
                    O[mt][3] *= f1;
                }
                // P^T as B fragment: via shared memory (the accumulator layout is the transpose of the B layout)
                __syncwarp();
                *(half2 *) &ps[w][r][hq]     = __floats2half2_rn(p0, p1);
                *(half2 *) &ps[w][r + 8][hq] = __floats2half2_rn(p2, p3);
                __syncwarp();
                uint32_t pb[2];
                flint_ldsm_x2_trans(pb, &ps[w][lane % 16][0]);
#pragma unroll
                for (int mt = 0; mt < FA_D/16; ++mt) {
                    // V^T: matrix j = lane/8 holds the dims 8*(j % 2) and the cells 8*(j / 2) of the tile
                    const int j = lane / 8;
                    uint32_t a[4];
                    flint_ldsm_x4_trans(a, &Vs[b][16*w + (j / 2)*8 + lane % 8][16*mt + (j % 2)*8]);
                    flint_mma(O[mt], a, pb[0], pb[1]);
                }
            }

            // buffer b is free once all warps are done with it
            if (i + 2 < n_loc) {
                __syncthreads();
                load_chunk(i + 2, b);
            }
        }

        // combine the states of the warps (in the order of the warps)
#pragma unroll
        for (int off = 4; off < WARP_SIZE; off *= 2) {
            L[0] += __shfl_xor_sync(0xFFFFFFFF, L[0], off);
            L[1] += __shfl_xor_sync(0xFFFFFFFF, L[1], off);
        }
        __syncthreads(); // all warps are done with the K buffers
        if (lane < 4) {
            Ms[w][hq]     = M[0];
            Ms[w][hq + 1] = M[1];
            Ls[w][hq]     = L[0];
            Ls[w][hq + 1] = L[1];
        }
#pragma unroll
        for (int mt = 0; mt < FA_D/16; ++mt) {
            Os[w][hq    ][16*mt + r]     = O[mt][0];
            Os[w][hq + 1][16*mt + r]     = O[mt][1];
            Os[w][hq    ][16*mt + r + 8] = O[mt][2];
            Os[w][hq + 1][16*mt + r + 8] = O[mt][3];
        }
        __syncthreads();

        Mt = nsp == 1 ? sinks_s[hh] : -INFINITY;
#pragma unroll
        for (int ww = 0; ww < FA_WARPS; ++ww) {
            Mt = fmaxf(Mt, Ms[ww][hh]);
        }
        Lt = sinks && nsp == 1 ? expf(sinks_s[hh] - Mt) : 0.0f;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            o[e] = 0.0f;
        }
#pragma unroll
        for (int ww = 0; ww < FA_WARPS; ++ww) {
            const float f = Ms[ww][hh] == -INFINITY ? 0.0f : expf(Ms[ww][hh] - Mt);
            Lt += f*Ls[ww][hh];
            const float4 oa = *(const float4 *) &Os[ww][hh][d0];
            const float4 ob = *(const float4 *) &Os[ww][hh][d0 + 4];
            o[0] += f*oa.x; o[1] += f*oa.y; o[2] += f*oa.z; o[3] += f*oa.w;
            o[4] += f*ob.x; o[5] += f*ob.y; o[6] += f*ob.z; o[7] += f*ob.w;
        }
        if (nsp > 1) {
            // several splits (all heads of the KV head in one block): the state of the split goes to part, which has nsp_max
            // states per token and KV head (tokens of other sequences in the batch may use more splits than this one)
            float * pp = part + ((int64_t) (t*gridDim.y + g)*nsp_max + sp)*FA_PART + hh*FA_HP;
            *(float4 *) (pp + d0)     = make_float4(o[0], o[1], o[2], o[3]);
            *(float4 *) (pp + d0 + 4) = make_float4(o[4], o[5], o[6], o[7]);
            if (tid % 16 == 0) {
                pp[FA_D]     = Mt;
                pp[FA_D + 1] = Lt;
            }
        }
    }
    if (n_mine == 0) {
        return; // the token needs fewer splits
    }

    if (nsp > 1) {
        // the first nc blocks of the KV head combine the states of the splits once all are written, block i the heads i,
        // i + nc, ... (all blocks of the grid are resident: one wave; the other blocks do not wait)
        const int sp0 = blockIdx.x / nhb;
        const int nc  = min(FA_GQA, min(nbs, nsp));
        unsigned int * written = counters + 2*(t*gridDim.y + g);
        unsigned int * waited  = written + 1;
        __syncthreads(); // the states of the splits of the block are written
        if (tid == 0) {
            ggml_cuda_red_add_release(written, (unsigned int) n_mine);
        }
        if (sp0 >= nc) {
            return;
        }
        if (tid == 0) {
            while (ggml_cuda_ld_acquire(written) < (unsigned int) nsp) {
            }
            // the last combining block to see all states resets the counters
            if (ggml_cuda_atom_add_acq_rel(waited, 1) == (unsigned int) nc - 1) {
                *written = 0;
                *waited  = 0;
            }
        }
        __syncthreads();

        // the states of head h of the splits in order, through shared memory in batches (the chunk buffers are free): all
        // copies of a batch are in flight at once. Thread d merges dim d
        static_assert(nthreads == FA_D && FA_HP % 4 == 0, "a thread per dim, 16 byte copies of whole states");
        constexpr int batch = 2*FA_STAGES*FA_CHUNK*FA_KSTR*sizeof(half) / (FA_HP*sizeof(float));
        float (*hs)[FA_HP] = (float (*)[FA_HP]) fa_smem;
        const float * ph = part + (int64_t) (t*gridDim.y + g)*nsp_max*FA_PART;
        for (int h = sp0; h < FA_GQA; h += nc) {
            float Mh = sinks_s[h];
            float Lh = sinks ? 1.0f : 0.0f;
            float oh = 0.0f;
            for (int i0 = 0; i0 < nsp; i0 += batch) {
                const int nb = min(batch, nsp - i0);
                __syncthreads(); // the previous batch is read
                for (int i = tid; i < nb*(FA_HP/4); i += nthreads) {
                    const int j = i / (FA_HP/4);
                    const int k = 4*(i % (FA_HP/4));
                    cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(&hs[j][k]), ph + (int64_t) (i0 + j)*FA_PART + h*FA_HP + k);
                }
                cp_async_wait_all();
                __syncthreads();
                for (int i = 0; i < nb; ++i) {
                    const float Mi = hs[i][FA_D];
                    const float Mn = fmaxf(Mh, Mi);
                    const float fo = Mh == -INFINITY ? 0.0f : expf(Mh - Mn);
                    const float fi = expf(Mi - Mn);
                    Mh = Mn;
                    Lh = Lh*fo + hs[i][FA_D + 1]*fi;
                    oh = oh*fo + fi*hs[i][tid];
                }
            }
            oh /= Lh;
            if (f32_out) {
                dst[(h0 + h)*FA_D + tid] = oh;
            }
            // q8_1 as quantize_q8_1: a warp per block
            const float  amax = warp_reduce_max(fabsf(oh));
            const float  d    = amax / 127.0f;
            const int8_t q    = amax == 0.0f ? 0 : roundf(oh / d);
            const int    sum  = warp_reduce_sum((int) q);
            block_q8_1 * yb   = yq + ((h0 + h)*FA_D)/QK8_1 + w;
            yb->qs[lane] = q;
            if (lane == 0) {
                yb->ds = make_half2(d, d * sum);
            }
        }
        return;
    }

    float amax = 0.0f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        o[e] /= Lt;
        amax = fmaxf(amax, fabsf(o[e]));
    }
    if (f32_out && hh < hpb) {
        *(float4 *) (dst + (h0 + hh)*FA_D + d0)     = make_float4(o[0], o[1], o[2], o[3]);
        *(float4 *) (dst + (h0 + hh)*FA_D + d0 + 4) = make_float4(o[4], o[5], o[6], o[7]);
    }

    // q8_1 as quantize_q8_1: 4 threads per block
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, 1, 4));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, 2, 4));
    const float d = amax / 127.0f;
    block_q8_1 * yb = yq + ((h0 + hh)*FA_D + d0)/QK8_1;
    int sum = 0;
    int8_t qv[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        qv[e] = amax == 0.0f ? 0 : roundf(o[e] / d);
        sum += qv[e];
    }
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 1, 4);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 2, 4);
    if (hh >= hpb) {
        return;
    }
    *(char4 *) (yb->qs + d0 % QK8_1)     = make_char4(qv[0], qv[1], qv[2], qv[3]);
    *(char4 *) (yb->qs + d0 % QK8_1 + 4) = make_char4(qv[4], qv[5], qv[6], qv[7]);
    if (tid % 4 == 0) {
        yb->ds = make_half2(d, d * sum);
    }
}

static bool flint_attn_rope_params(const ggml_tensor * rope, int & mode, flint_attn_rope & rp) {
    const int32_t * op = (const int32_t *) rope->op_params;
    const int n_dims     = op[1];
    const int rmode      = op[2];
    const int n_ctx_orig = op[4];
    float freq_base, beta_fast, beta_slow;
    memcpy(&freq_base,      op +  5, sizeof(float));
    memcpy(&rp.freq_scale,  op +  6, sizeof(float));
    memcpy(&rp.ext_factor,  op +  7, sizeof(float));
    memcpy(&rp.attn_factor, op +  8, sizeof(float));
    memcpy(&beta_fast,      op +  9, sizeof(float));
    memcpy(&beta_slow,      op + 10, sizeof(float));
    if (n_dims != FA_D || op[15] != 0 || rope->src[2] != nullptr || rope->src[1]->type != GGML_TYPE_I32 ||
            (rmode != GGML_ROPE_TYPE_NORMAL && rmode != GGML_ROPE_TYPE_NEOX)) {
        return false;
    }
    mode = rmode == GGML_ROPE_TYPE_NEOX ? 2 : 1;
    rp.theta_scale = powf(freq_base, -2.0f/n_dims);
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, ggml_rope_get_truncate(rope), rp.corr_dims);
    return true;
}

// shared memory for n_kv cells
static size_t flint_attn_smem(const int n_kv) {
    return 2*FA_STAGES*FA_CHUNK*FA_KSTR*sizeof(half) + (size_t) n_kv*sizeof(half);
}

static bool flint_overlap(const ggml_tensor * a, const int64_t na, const ggml_tensor * b, const int64_t nb) {
    const char * a0 = (const char *) a->data;
    const char * b0 = (const char *) b->data;
    return a0 < b0 + nb && b0 < a0 + na;
}

static bool flint_attn_row_ok(const ggml_tensor * t, const int64_t nrows, const int64_t ntok) {
    return t->type == GGML_TYPE_F32 && t->ne[0] == FA_D && t->ne[1] == nrows && t->ne[2] == ntok && t->ne[3] == 1 &&
        t->nb[0] == sizeof(float) && t->nb[1] % 16 == 0 && t->nb[2] % 16 == 0 && ggml_cuda_is_aligned(t, 16);
}

// size of the memory a view spans
static int64_t flint_span(const ggml_tensor * t) {
    int64_t n = ggml_type_size(t->type);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        n += (t->ne[i] - 1)*t->nb[i];
    }
    return n;
}

bool ggml_cuda_flint_attn_supported(ggml_backend_cuda_context & ctx, const ggml_cuda_flint_attn_args & a) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_AMPERE) {
        return false;
    }
    const ggml_tensor * fa = a.fa;
    const ggml_tensor * Q  = fa->src[0];
    const ggml_tensor * K  = fa->src[1];
    const ggml_tensor * V  = fa->src[2];
    const ggml_tensor * mask = fa->src[3];
    const ggml_tensor * sinks = fa->src[4];
    const int64_t n_head_kv = K->ne[2];
    const int64_t n_kv      = K->ne[1];

    float max_bias, softcap;
    memcpy(&max_bias, (const float *) fa->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) fa->op_params + 2, sizeof(float));

    if (fa->type != GGML_TYPE_F32 || !ggml_is_contiguous(fa) || fa->ne[0] != FA_D || fa->ne[1] != Q->ne[2] || fa->ne[2] != Q->ne[1] || fa->ne[3] != Q->ne[3] || !ggml_cuda_is_aligned(fa, 16) ||
            Q->ne[0] != FA_D || Q->ne[1] > FLINT_MAX_TOKENS || Q->ne[1]*Q->ne[3] > FLINT_ATTN_MAX_TOKENS || Q->ne[2] != FA_GQA*n_head_kv ||
            n_head_kv > FA_MAX_KV_HEADS || n_kv > FA_MAX_KV || max_bias != 0.0f || softcap != 0.0f ||
            fa->src[5] != nullptr || mask == nullptr || mask->type != GGML_TYPE_F16 || mask->ne[0] != n_kv ||
            mask->ne[1] < Q->ne[1] || mask->ne[2] != 1 || (mask->ne[3] != 1 && mask->ne[3] != Q->ne[3]) ||
            mask->nb[1] % 16 != 0 || mask->nb[3] % 16 != 0 ||
            (sinks && (sinks->type != GGML_TYPE_F32 || sinks->ne[0] != Q->ne[2] || !ggml_is_contiguous(sinks)))) {
        return false;
    }
    const int64_t ntok = Q->ne[1]*Q->ne[3];
    // K/V: F16 cache views [D, n_kv, n_head_kv, n_stream], heads adjacent within a cell, streams of the same cell count
    for (const ggml_tensor * t : { K, V }) {
        if (t->type != GGML_TYPE_F16 || t->ne[0] != FA_D || t->ne[2] != n_head_kv || t->ne[3] != Q->ne[3] ||
                t->nb[0] != sizeof(half) || t->nb[2] != FA_D*sizeof(half) || t->nb[1] % 16 != 0 ||
                t->nb[3] % t->nb[1] != 0 || t->nb[3] != K->nb[3]/K->nb[1]*t->nb[1] || !ggml_cuda_is_aligned(t, 16)) {
            return false;
        }
    }
    // the SET_ROWS write one row of all heads per token into the cache rows of the K/V views (from the start of the cache)
    for (const auto & [set, kv] : { std::make_pair(a.k_set, K), std::make_pair(a.v_set, V) }) {
        const char * base = (const char *) set->data;
        const char * view = (const char *) kv->data;
        if (set->type != GGML_TYPE_F16 || view < base || (view - base) % kv->nb[1] != 0 || set->nb[1] != kv->nb[1] ||
                set->ne[0] != FA_D*n_head_kv || set->src[1]->type != GGML_TYPE_I64 || !ggml_is_contiguous(set->src[1]) ||
                ggml_nelements(set->src[1]) != ntok || ggml_nelements(set->src[0]) != FA_D*n_head_kv*ntok) {
            return false;
        }
    }
    if (!flint_attn_row_ok(a.q, Q->ne[2], ntok) || !flint_attn_row_ok(a.k, n_head_kv, ntok) || !flint_attn_row_ok(a.v, n_head_kv, ntok)) {
        return false;
    }
    if (a.q_rope && ggml_nelements(a.q_rope->src[1]) < ntok) {
        return false;
    }
    for (const ggml_tensor * t : { a.q_norm_w, a.k_norm_w }) {
        if (t->type != GGML_TYPE_F32 || ggml_nelements(t) != FA_D || !ggml_is_contiguous(t) || !ggml_cuda_is_aligned(t, 16)) {
            return false;
        }
    }
    if ((a.q_rope == nullptr) != (a.k_rope == nullptr)) {
        return false;
    }
    if (a.q_rope) {
        int mode_q, mode_k;
        flint_attn_rope rq, rk;
        if (!flint_attn_rope_params(a.q_rope, mode_q, rq) || !flint_attn_rope_params(a.k_rope, mode_k, rk) ||
                a.q_rope->src[1] != a.k_rope->src[1] || memcmp(a.q_rope->op_params, a.k_rope->op_params, sizeof(a.q_rope->op_params)) != 0) {
            return false;
        }
    }
    // dst may reuse the memory of Q (each block reads the Q rows of its heads before writing their outputs), not of K/V
    // (with several tokens only if Q is laid out as dst, or dst is not written)
    const int64_t n_head = Q->ne[2];
    const int64_t q_size = flint_span(a.q);
    const int64_t k_size = flint_span(a.k);
    const int64_t v_size = flint_span(a.v);
    const int64_t d_size = ggml_nbytes(fa);
    if (a.f32_out && (flint_overlap(fa, d_size, a.k, k_size) || flint_overlap(fa, d_size, a.v, v_size) ||
            (flint_overlap(fa, d_size, a.q, q_size) && (fa->data != a.q->data || a.q->nb[1] != FA_D*sizeof(float) ||
                                                        (ntok > 1 && a.q->nb[2] != n_head*FA_D*sizeof(float)))))) {
        return false;
    }
    return n_kv % 8 == 0 && ggml_cuda_is_aligned(mask, 16) && ctx.fused_scratch_ready();
}

// splits of the cells: as many blocks of one token as can be resident (one wave), at most FA_MAX_CPS chunks per split. They
// do not depend on the number of tokens (a token gets the same result in batches of any size); the kernel uses fewer for a
// token with fewer chunks with unmasked cells.
static int flint_attn_splits(const int n_kv, const int n_blocks, const int max_blocks) {
    const int n_chunks = (n_kv + FA_CHUNK - 1) / FA_CHUNK;
    if (n_chunks < FA_MIN_SPLIT_CHUNKS) {
        return 1;
    }
    return std::max({1, std::min(n_chunks, max_blocks / n_blocks), (n_chunks + FA_MAX_CPS - 1) / FA_MAX_CPS});
}

void ggml_cuda_flint_attn(ggml_backend_cuda_context & ctx, const ggml_cuda_flint_attn_args & a) {
    const ggml_tensor * fa = a.fa;
    const ggml_tensor * K  = fa->src[1];
    const ggml_tensor * V  = fa->src[2];
    const int n_head_kv = K->ne[2];
    const int n_kv      = K->ne[1];
    const ggml_tensor * Q    = fa->src[0];
    const ggml_tensor * mask = fa->src[3];
    const int ne1  = (int) Q->ne[1];
    const int ntok = (int) (Q->ne[1]*Q->ne[3]);

    float kq_scale;
    memcpy(&kq_scale, fa->op_params, sizeof(float));

    int mode = 0;
    flint_attn_rope rp = {};
    const int32_t * pos = nullptr;
    if (a.q_rope) {
        flint_attn_rope_params(a.q_rope, mode, rp);
        pos = (const int32_t *) a.q_rope->src[1]->data;
    }

    static_assert(FA_GQA*FA_MAX_KV_HEADS*FA_D <= ggml_backend_cuda_context::q8_1_max_ncols, "q8_1 buffer too small");
    block_q8_1 * yq = (block_q8_1 *) ctx.q8_1_next_buf();
    const int64_t k_row0 = ((const char *) K->data - (const char *) a.k_set->data) / K->nb[1];
    const int64_t v_row0 = ((const char *) V->data - (const char *) a.v_set->data) / V->nb[1];
    const auto launch = [&](auto kernel) {
        constexpr int nthreads = FA_WARPS*WARP_SIZE;
        static int max_blocks[GGML_CUDA_MAX_DEVICES][3] = {};
        int & mb = max_blocks[ctx.device][mode];
        if (!mb) {
            CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, flint_attn_smem(FA_MAX_KV)));
            CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
            int per_sm = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, nthreads, flint_attn_smem(FA_MAX_KV)));
            mb = std::max(1, per_sm)*ggml_cuda_info().devices[ctx.device].nsm;
        }
        const int n_split = flint_attn_splits(n_kv, n_head_kv, mb);
        // without splits, blocks of FA_HPB_1 heads (more blocks: the work outside the chunks is latency bound)
        const int hpb = n_split == 1 ? FA_HPB_1 : FA_GQA;
        // blocks per KV head: with several tokens a block may process several splits (one wave)
        const int nbs = std::min(n_split, std::max(1, mb / (n_head_kv*ntok)));
        ggml_cuda_pool_alloc<float> part(ctx.pool(), n_split > 1 ? (size_t) ntok*n_head_kv*n_split*FA_PART : 1);
        unsigned int * counters = (unsigned int *) ((char *) ctx.get_fused_scratch() + fa_counters_offset);
        const ggml_cuda_kernel_launch_params params(dim3(FA_GQA/hpb*nbs, n_head_kv, ntok), dim3(nthreads, 1, 1), flint_attn_smem(n_kv), ctx.stream());
        ggml_cuda_kernel_launch(kernel, params,
            (const float *) a.q->data, (int64_t) (a.q->nb[1]/sizeof(float)), (int64_t) (a.q->nb[2]/sizeof(float)),
            (const float *) a.k->data, (int64_t) (a.k->nb[1]/sizeof(float)), (int64_t) (a.k->nb[2]/sizeof(float)),
            (const float *) a.v->data, (int64_t) (a.v->nb[1]/sizeof(float)), (int64_t) (a.v->nb[2]/sizeof(float)),
            (const float *) a.q_norm_w->data, (const float *) a.k_norm_w->data, a.eps_q, a.eps_k,
            pos, rp,
            (half *) K->data, (half *) V->data, (int64_t) (K->nb[1]/sizeof(half)), (int64_t) (V->nb[1]/sizeof(half)),
            (int64_t) (K->nb[3]/sizeof(half)), (int64_t) (V->nb[3]/sizeof(half)),
            (const int64_t *) a.k_set->src[1]->data, (const int64_t *) a.v_set->src[1]->data, k_row0, v_row0,
            (int64_t) (K->nb[3]/K->nb[1]),
            (const half *) mask->data, (int64_t) (mask->nb[1]/sizeof(half)), mask->ne[3] == 1 ? 0 : (int64_t) (mask->nb[3]/sizeof(half)),
            ne1, n_kv, n_split, hpb, fa->src[4] ? (const float *) fa->src[4]->data : nullptr, kq_scale,
            (float *) fa->data, a.f32_out, yq, part.get(), counters);
    };
    switch (mode) {
        case 0: launch(flint_attn_kernel<0>); break;
        case 1: launch(flint_attn_kernel<1>); break;
        default: launch(flint_attn_kernel<2>); break;
    }
    if (a.fa_2d) {
        ctx.q8_1_set(a.fa_2d);
    }
}
