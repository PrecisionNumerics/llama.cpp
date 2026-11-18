#include "flint-fused.cuh"
#include "cp-async.cuh"
#include "vecdotq.cuh"
#include "quantize.cuh"
#include "unary.cuh"

// Router: FR_BLOCKS blocks of FR_WARPS warps. Every block normalizes the activations, then FR_WPR warps compute one router
// logit (BF16 weights, F32 activations, the weights are loaded before the PDL wait). The last block to finish (counter in
// scratch) runs the sigmoid + bias top-k (as topk_moe_cuda) on the logits and resets the counter. With defer, the top-k is
// left to the next kernel that reads the ids (see ggml_cuda_topk_deferred). Block 0 also writes the normalized activations as F32 and q8_1.
// The results are bit-exact with rms_norm_mul_q8_1_f32 (256 threads), mul_mat_vec_f (256 threads) and topk_moe_cuda:
// each thread does the work of FR_NH threads of those kernels, and the 8 warp sums are added in the order of their final
// xor butterfly.

#define FR_NEXPERT FLINT_NEXPERT
#define FR_NCOLS   2048
#define FR_WARPS   8
#define FR_NH      (256 / (FR_WARPS * WARP_SIZE)) // reference threads per thread
#define FR_THREADS (FR_WARPS * WARP_SIZE)
#define FR_WPR     1 // warps per router row
#define FR_ROWS    (FR_WARPS / FR_WPR)
#define FR_BLOCKS  (FR_NEXPERT / FR_ROWS)

static_assert(FR_THREADS >= FLINT_NEXPERT, "the last block runs flint_topk_block");
static_assert(FR_NCOLS == 2 * 4 * 256, "the reference kernels use 256 threads with 2 float4 or 4 bf16 pairs each");

struct flint_route_scratch {
    unsigned int counter;
    float        logits[FR_NEXPERT];
};

// xor butterfly result of 8 values on lanes 0..7 (the other lanes 0)
static __device__ __forceinline__ float flint_sum8(const float * v) {
    return ((v[0] + v[4]) + (v[2] + v[6])) + ((v[1] + v[5]) + (v[3] + v[7]));
}

static __global__ void __launch_bounds__(FR_THREADS) flint_route_kernel(
        const float * x, const float * norm_w, float * xn, block_q8_1 * yq,
        const char * router, const int64_t router_nb1, const float * bias, flint_route_scratch * scratch,
        int32_t * ids, float * weights, const int n_used, const float eps, const float clamp_val,
        const float scale_val, const bool with_norm, const bool defer) {
    __shared__ float4 xs[FR_NCOLS / 4];
    __shared__ float  red[8];
    __shared__ float  part[FR_WARPS][8];
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
    constexpr int nvw = 8 / FR_WPR;
    const int rl  = warp / FR_WPR; // row of this block
    const int sub = warp % FR_WPR;
    const int row = blockIdx.x * FR_ROWS + rl;
    const nv_bfloat162 * w2 = (const nv_bfloat162 *) (router + row * router_nb1);
    nv_bfloat162 wv[nvw][4];
#pragma unroll
    for (int j = 0; j < nvw; ++j) {
#pragma unroll
        for (int it = 0; it < 4; ++it) {
            wv[j][it] = w2[(sub * nvw + j) * WARP_SIZE + lane + it * 256];
        }
    }

    const float bias_v = defer ? 0.0f : flint_topk_bias(bias);

    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();

    float4 v[FR_NH][2];
#pragma unroll
    for (int h = 0; h < FR_NH; ++h) {
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            v[h][k] = ((const float4 *) x)[h * FR_THREADS + tid + 256 * k];
            sum += v[h][k].x*v[h][k].x + v[h][k].y*v[h][k].y + v[h][k].z*v[h][k].z + v[h][k].w*v[h][k].w;
        }
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            red[h * FR_WARPS + warp] = sum;
        }
    }
    __syncthreads();
    const float scale = rsqrtf(flint_sum8(red) / FR_NCOLS + eps);

#pragma unroll
    for (int h = 0; h < FR_NH; ++h) {
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int i = h * FR_THREADS + tid + 256 * k;
            float4 r = make_float4(scale*v[h][k].x, scale*v[h][k].y, scale*v[h][k].z, scale*v[h][k].w);
            r.x *= m[h][k].x;
            r.y *= m[h][k].y;
            r.z *= m[h][k].z;
            r.w *= m[h][k].w;
            xs[i] = r;
            if (blockIdx.x == 0) {
                ((float4 *) xn)[i] = r;

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
                block_q8_1 & yb = yq[i / 8];
                ((char4 *) yb.qs)[i % 8] = q;
                if (i % 8 == 0) {
                    yb.ds = make_half2(d, d * qsum);
                }
            }
        }
    }
    __syncthreads();

    // mul_mat_vec_f: reference warp vw, lane l sums the pairs vw*32 + l + 256*it in order; a warp does 8/FR_WPR of them
    const float2 * y2 = (const float2 *) xs;
#pragma unroll
    for (int j = 0; j < nvw; ++j) {
        const int vw = sub * nvw + j;
        float a = 0.0f;
#pragma unroll
        for (int it = 0; it < 4; ++it) {
            const float2 tmpx = __bfloat1622float2(wv[j][it]);
            const float2 tmpy = y2[vw * WARP_SIZE + lane + it * 256];
            ggml_cuda_mad(a, tmpx, tmpy);
        }
        a = warp_reduce_sum(a);
        if (lane == 0) {
            part[rl][vw] = a;
        }
    }
    __syncthreads();
    if (sub == 0 && lane == 0) {
        scratch->logits[row] = flint_sum8(part[rl]);
    }
    if (defer) {
        return;
    }

    // last block: all logits are written
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        last = atomicAdd(&scratch->counter, 1) == FR_BLOCKS - 1;
    }
    __syncthreads();
    if (!last) {
        return;
    }
    __threadfence();

    __shared__ int   ids_s[FLINT_MAX_USED];
    __shared__ float w_s[FLINT_MAX_USED];
    flint_topk_block(scratch->logits, bias_v, n_used, with_norm, clamp_val, scale_val, ids_s, w_s);
    if (tid == 0) {
        scratch->counter = 0;
    }
    if (tid < n_used) {
        ids[tid]     = ids_s[tid];
        weights[tid] = w_s[tid];
    }
}

// RMS_NORM -> MUL -> MUL_MAT (Q4_0, FR_NCOLS columns): FN_WARPS warps, each 4 warps do FN_RPG rows. Each block normalizes
// and quantizes the activations: threads 0..255 do the work of rms_norm_mul_q8_1_f32 (256 threads). The dot products are those
// of mul_mat_vec_q for ncols_dst 1 (4 warps per row; thread t of a group: q4_0 block t/2, ints 2*(t%2), 2*(t%2) + 1).
// Few large blocks: every block reads the activations. The weights are loaded before the PDL wait.
// With a gate matrix vg (glu): the SWIGLU of mul_mat_vec_q with the fused gate (dst = up * silu(gate), vx: up), FN_GLU_RPG
// rows per group (grid in one wave with 2 blocks per SM).
#define FN_WARPS   16
#define FN_MM_RPG  4
#define FN_GLU_RPG 5

template <int FN_RPG_T, bool glu>
static __global__ void __launch_bounds__(FN_WARPS * WARP_SIZE) flint_norm_mmvq_kernel(
        const float * x, const float * norm_w, const float eps, const char * vx, const char * vg, const int64_t stride_row,
        const int nrows, float * dst) {
    constexpr int FN_RPG    = FN_RPG_T;
    constexpr int FN_GROUPS = FN_WARPS / 4;
    constexpr int FN_ROWS   = FN_RPG * FN_GROUPS;
    static_assert(FR_NCOLS / QK4_0 == 4 * WARP_SIZE / 2, "one q4_0 block per 2 threads of a group");
    static_assert(FN_WARPS * WARP_SIZE >= 256, "256 threads normalize");
    __shared__ block_q8_1 ys[FR_NCOLS / QK8_1];
    __shared__ float      red[8];
    __shared__ float      part[FN_GROUPS][3][FN_RPG][WARP_SIZE];
    __shared__ float      part_g[glu ? FN_GROUPS : 1][3][glu ? FN_RPG : 1][WARP_SIZE];

    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int grp  = warp / 4;
    const int gw   = warp % 4;
    const int row0 = blockIdx.x * FN_ROWS + grp * FN_RPG;
    const int kbx  = (tid % 128) / 2;
    const int kqs  = 2 * (tid % 2);
    const bool nt  = tid < 256; // normalizing thread: float4 tid + 256*k

    ggml_cuda_pdl_lc();

    float4 m[2];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        m[k] = nt ? ((const float4 *) norm_w)[tid + 256 * k] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
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

    float4 xv[2];
    if (nt) {
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            xv[k] = ((const float4 *) x)[tid + 256 * k];
            sum += xv[k].x*xv[k].x + xv[k].y*xv[k].y + xv[k].z*xv[k].z + xv[k].w*xv[k].w;
        }
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            red[warp] = sum;
        }
    }
    __syncthreads();
    const float scale = rsqrtf(flint_sum8(red) / FR_NCOLS + eps);

    if (nt) {
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int i = tid + 256 * k;
            float4 r = make_float4(scale*xv[k].x, scale*xv[k].y, scale*xv[k].z, scale*xv[k].w);
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
            block_q8_1 & yb = ys[i / 8];
            ((char4 *) yb.qs)[i % 8] = q;
            if (i % 8 == 0) {
                yb.ds = make_half2(d, d * qsum);
            }
        }
    }
    __syncthreads();

    int u[4];
    u[0] = get_int_b4(ys[kbx].qs, kqs + 0);
    u[1] = get_int_b4(ys[kbx].qs, kqs + 0 + QI4_0);
    u[2] = get_int_b4(ys[kbx].qs, kqs + 1);
    u[3] = get_int_b4(ys[kbx].qs, kqs + 1 + QI4_0);
    float tmp[FN_RPG];
    float tmp_g[glu ? FN_RPG : 1];
#pragma unroll
    for (int i = 0; i < FN_RPG; ++i) {
        tmp[i] = 0.0f;
        tmp[i] += vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v[i], u, d4[i], ys[kbx].ds);
        if constexpr (glu) {
            tmp_g[i] = 0.0f;
            tmp_g[i] += vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(vgt[i], u, dg4[i], ys[kbx].ds);
        }
    }

    if (gw > 0) {
#pragma unroll
        for (int i = 0; i < FN_RPG; ++i) {
            part[grp][gw - 1][i][lane] = tmp[i];
            if constexpr (glu) {
                part_g[grp][gw - 1][i][lane] = tmp_g[i];
            }
        }
    }
    __syncthreads();
    if (gw > 0) {
        return;
    }
#pragma unroll
    for (int i = 0; i < FN_RPG; ++i) {
#pragma unroll
        for (int l = 0; l < 3; ++l) {
            tmp[i] += part[grp][l][i][lane];
            if constexpr (glu) {
                tmp_g[i] += part_g[grp][l][i][lane];
            }
        }
        tmp[i] = warp_reduce_sum(tmp[i]);
        if constexpr (glu) {
            tmp_g[i] = warp_reduce_sum(tmp_g[i]);
        }
        if (lane == i && row0 + i < nrows) {
            float result = tmp[i];
            if constexpr (glu) {
                // as mul_mat_vec_q with the fused gate (zero biases added)
                result += 0.0f;
                float gate_value = tmp_g[i];
                gate_value += 0.0f;
                result *= ggml_cuda_op_silu_single(gate_value);
            }
            dst[row0 + i] = result;
        }
    }
}

bool ggml_cuda_flint_norm_mmvq_supported(const ggml_tensor * x, const ggml_tensor * norm_w, const ggml_tensor * mm) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_AMPERE) {
        return false;
    }
    const ggml_tensor * w = mm->src[0];
    return x->type == GGML_TYPE_F32 && x->ne[0] == FR_NCOLS && ggml_nrows(x) == 1 && ggml_is_contiguous(x) &&
        ggml_cuda_is_aligned(x, 16) &&
        norm_w->type == GGML_TYPE_F32 && norm_w->ne[0] == FR_NCOLS && ggml_nrows(norm_w) == 1 &&
        ggml_is_contiguous(norm_w) && ggml_cuda_is_aligned(norm_w, 16) &&
        w->type == GGML_TYPE_Q4_0 && w->ne[0] == FR_NCOLS && w->ne[2] == 1 && w->ne[3] == 1 &&
        w->nb[1] % sizeof(block_q4_0) == 0 && w->ne[1] < INT_MAX &&
        mm->type == GGML_TYPE_F32 && mm->ne[1] == 1 && ggml_is_contiguous(mm);
}

void ggml_cuda_flint_norm_mmvq(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * norm_w,
        const float eps, ggml_tensor * mm) {
    const ggml_tensor * w = mm->src[0];
    const int nrows = (int) w->ne[1];
    constexpr int rows = FN_MM_RPG * FN_WARPS / 4;
    const ggml_cuda_kernel_launch_params params(dim3((nrows + rows - 1) / rows), dim3(FN_WARPS * WARP_SIZE), 0, ctx.stream());
    ggml_cuda_kernel_launch(flint_norm_mmvq_kernel<FN_MM_RPG, false>, params, (const float *) x->data, (const float *) norm_w->data, eps,
        (const char *) w->data, (const char *) nullptr, (int64_t) (w->nb[1] / sizeof(block_q4_0)), nrows, (float *) mm->data);
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
        glu->type == GGML_TYPE_F32 && ggml_is_contiguous(glu) && ggml_nrows(glu) == 1 && glu->ne[0] == wu->ne[1];
}

void ggml_cuda_flint_norm_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * norm_w,
        const float eps, const ggml_tensor * up, const ggml_tensor * gate, ggml_tensor * glu) {
    const ggml_tensor * wu = up->src[0];
    const ggml_tensor * wg = gate->src[0];
    const int nrows = (int) wu->ne[1];
    constexpr int rows = FN_GLU_RPG * FN_WARPS / 4;
    const ggml_cuda_kernel_launch_params params(dim3((nrows + rows - 1) / rows), dim3(FN_WARPS * WARP_SIZE), 0, ctx.stream());
    ggml_cuda_kernel_launch(flint_norm_mmvq_kernel<FN_GLU_RPG, true>, params, (const float *) x->data, (const float *) norm_w->data, eps,
        (const char *) wu->data, (const char *) wg->data, (int64_t) (wu->nb[1] / sizeof(block_q4_0)), nrows, (float *) glu->data);
}

// MUL_MAT (Q4_0) + ADD: FO_WARPS warps, each 4 warps do FO_RPG rows, every row as one mul_mat_vec_q block (4 warps, 1 row:
// thread t sums items t, t + 128, ... in order (item i: q4_0 block i/2, ints 2*(i%2), 2*(i%2) + 1), then warps 1..3 are added
// to warp 0 in order and the warp is reduced; the residual is added last). The weights are loaded before the PDL wait.
#define FO_WARPS 8
#define FO_RPG   2
#define FO_GROUPS (FO_WARPS / 4)

template <int ipt> // items per thread
static __global__ void __launch_bounds__(FO_WARPS * WARP_SIZE) flint_mmvq_add_kernel(
        const char * vx, const int64_t stride_row, const block_q8_1 * y, const float * res, const int nrows, float * dst) {
    constexpr int nblk = ipt * 64; // q4_0 blocks per row
    __shared__ block_q8_1 ys[nblk];
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

    static_assert(sizeof(ys) % 16 == 0, "ys is copied as int4");
    for (int i = tid; i < (int) (sizeof(ys) / 16); i += FO_WARPS * WARP_SIZE) {
        ((int4 *) ys)[i] = ((const int4 *) y)[i];
    }
    const float rv = gw == 0 && lane < FO_RPG && row0 + lane < nrows ? res[row0 + lane] : 0.0f;
    __syncthreads();

    float tmp[FO_RPG];
#pragma unroll
    for (int r = 0; r < FO_RPG; ++r) {
        tmp[r] = 0.0f;
    }
#pragma unroll
    for (int j = 0; j < ipt; ++j) {
        const int item = gt + 128 * j;
        const block_q8_1 & yb = ys[item / 2];
        const int kqs = 2 * (item % 2);
        int u[4];
        u[0] = get_int_b4(yb.qs, kqs + 0);
        u[1] = get_int_b4(yb.qs, kqs + 0 + QI4_0);
        u[2] = get_int_b4(yb.qs, kqs + 1);
        u[3] = get_int_b4(yb.qs, kqs + 1 + QI4_0);
#pragma unroll
        for (int r = 0; r < FO_RPG; ++r) {
            tmp[r] += vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v[r][j], u, d4[r][j], yb.ds);
        }
    }

    if (gw > 0) {
#pragma unroll
        for (int r = 0; r < FO_RPG; ++r) {
            part[grp][gw - 1][r][lane] = tmp[r];
        }
    }
    __syncthreads();
    if (gw > 0) {
        return;
    }
#pragma unroll
    for (int r = 0; r < FO_RPG; ++r) {
#pragma unroll
        for (int l = 0; l < 3; ++l) {
            tmp[r] += part[grp][l][r][lane];
        }
        tmp[r] = warp_reduce_sum(tmp[r]);
        if (lane == r && row0 + r < nrows) {
            dst[row0 + r] = tmp[r] + rv;
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
    return w->type == GGML_TYPE_Q4_0 && w->ne[0] % (QK4_0 * 64) == 0 && ipt >= 1 && ipt <= 4 && w->ne[2] == 1 &&
        w->ne[3] == 1 && w->ne[1] < INT_MAX && w->nb[1] % sizeof(block_q4_0) == 0 &&
        x->type == GGML_TYPE_F32 && ggml_nrows(x) == 1 && ggml_is_contiguous(x) &&
        (ctx.q8_1_lookup(x) != nullptr || (x->ne[0] % MATRIX_ROW_PADDING == 0 && x->ne[0] <= ggml_backend_cuda_context::q8_1_max_ncols)) &&
        mm->type == GGML_TYPE_F32 && ggml_nrows(mm) == 1 &&
        res->type == GGML_TYPE_F32 && ggml_are_same_shape(res, mm) && ggml_is_contiguous(res) &&
        add->type == GGML_TYPE_F32 && ggml_are_same_shape(add, mm) && ggml_is_contiguous(add);
}

void ggml_cuda_flint_mmvq_add(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, ggml_tensor * add) {
    const ggml_tensor * w   = mm->src[0];
    const ggml_tensor * res = add->src[0] == mm ? add->src[1] : add->src[0];
    const int nrows = (int) w->ne[1];
    const int ipt   = (int) (w->ne[0] / (QK4_0 * 64));
    const ggml_tensor * x = mm->src[1];
    if (ctx.q8_1_lookup(x) == nullptr) {
        // as mul_mat_vec_q quantizes it (the weights of this kernel are loaded during the quantization)
        void * q = ctx.q8_1_next_buf();
        quantize_row_q8_1_cuda((const float *) x->data, q, x->ne[0], x->ne[0], x->ne[0], x->ne[0], x->ne[0], 1, 1, 1, ctx.stream());
        ctx.q8_1_set(x);
    }
    const block_q8_1 * y = (const block_q8_1 *) ctx.q8_1_lookup(x);
    const ggml_cuda_kernel_launch_params params(dim3((nrows + FO_GROUPS * FO_RPG - 1) / (FO_GROUPS * FO_RPG)),
        dim3(FO_WARPS * WARP_SIZE), 0, ctx.stream());
    const char *  vx = (const char *) w->data;
    const int64_t sr = w->nb[1] / sizeof(block_q4_0);
    switch (ipt) {
        case 1:  ggml_cuda_kernel_launch(flint_mmvq_add_kernel<1>, params, vx, sr, y, (const float *) res->data, nrows, (float *) add->data); break;
        case 2:  ggml_cuda_kernel_launch(flint_mmvq_add_kernel<2>, params, vx, sr, y, (const float *) res->data, nrows, (float *) add->data); break;
        case 3:  ggml_cuda_kernel_launch(flint_mmvq_add_kernel<3>, params, vx, sr, y, (const float *) res->data, nrows, (float *) add->data); break;
        default: ggml_cuda_kernel_launch(flint_mmvq_add_kernel<4>, params, vx, sr, y, (const float *) res->data, nrows, (float *) add->data); break;
    }
}

bool ggml_cuda_flint_route_supported(ggml_backend_cuda_context & ctx, const ggml_cuda_flint_route_args & a) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_AMPERE) {
        return false;
    }
    return a.x->type == GGML_TYPE_F32 && a.x->ne[0] == FR_NCOLS && ggml_nrows(a.x) == 1 && ggml_is_contiguous(a.x) &&
        a.norm_w->type == GGML_TYPE_F32 && a.norm_w->ne[0] == FR_NCOLS && ggml_nrows(a.norm_w) == 1 &&
        a.xn->type == GGML_TYPE_F32 && ggml_is_contiguous(a.xn) &&
        a.router->type == GGML_TYPE_BF16 && a.router->ne[0] == FR_NCOLS && a.router->ne[1] == FR_NEXPERT &&
        a.router->nb[1] % 16 == 0 &&
        a.bias != nullptr && a.bias->type == GGML_TYPE_F32 && a.bias->ne[0] == FR_NEXPERT && ggml_is_contiguous(a.bias) &&
        a.ids->type == GGML_TYPE_I32 && a.ids->ne[0] == ggml_nelements(a.weights) && ggml_nrows(a.ids) == 1 &&
        a.weights->type == GGML_TYPE_F32 && ggml_nelements(a.weights) <= FLINT_MAX_USED && ggml_is_contiguous(a.weights) &&
        ggml_cuda_is_aligned(a.x, 16) && ggml_cuda_is_aligned(a.norm_w, 16) && ggml_cuda_is_aligned(a.xn, 16) &&
        ggml_cuda_is_aligned(a.router, 16) && ctx.fused_scratch_ready();
}

void ggml_cuda_flint_route(ggml_backend_cuda_context & ctx, const ggml_cuda_flint_route_args & a) {
    static_assert(FR_NCOLS <= ggml_backend_cuda_context::q8_1_max_ncols, "q8_1 buffer too small");
    block_q8_1 * yq = (block_q8_1 *) ctx.q8_1_next_buf();
    static_assert(sizeof(flint_route_scratch) <= ggml_backend_cuda_context::fused_scratch_size, "scratch too small");
    flint_route_scratch * scratch = (flint_route_scratch *) ctx.get_fused_scratch();
    const ggml_cuda_kernel_launch_params params(dim3(FR_BLOCKS, 1, 1), dim3(FR_THREADS, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(flint_route_kernel, params,
        (const float *) a.x->data, (const float *) a.norm_w->data, (float *) a.xn->data, yq,
        (const char *) a.router->data, (int64_t) a.router->nb[1], (const float *) a.bias->data, scratch,
        (int32_t *) a.ids->data, (float *) a.weights->data, (int) ggml_nelements(a.weights), a.eps, a.clamp, a.scale,
        a.norm, a.defer);
    ctx.q8_1_set(a.xn);
    if (a.defer) {
        ggml_cuda_topk_deferred & d = ctx.topk_deferred;
        d.ids_src = a.ids->data;
        d.logits  = scratch->logits;
        d.bias    = (const float *) a.bias->data;
        d.ids     = (int32_t *) a.ids->data;
        d.weights = (float *) a.weights->data;
        d.n_used  = (int) ggml_nelements(a.weights);
        d.norm    = a.norm;
        d.clamp   = a.clamp;
        d.scale   = a.scale;
    }
}

static __global__ void __launch_bounds__(FLINT_NEXPERT) flint_topk_kernel(const ggml_cuda_topk_deferred d) {
    const float bias_v = flint_topk_bias(d.bias);
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    __shared__ int   ids_s[FLINT_MAX_USED];
    __shared__ float w_s[FLINT_MAX_USED];
    flint_topk_block(d.logits, bias_v, d.n_used, d.norm, d.clamp, d.scale, ids_s, w_s);
    if ((int) threadIdx.x < d.n_used) {
        d.ids[threadIdx.x]     = ids_s[threadIdx.x];
        d.weights[threadIdx.x] = w_s[threadIdx.x];
    }
}

void ggml_cuda_flint_topk_flush(ggml_backend_cuda_context & ctx) {
    if (ctx.topk_deferred.ids_src == nullptr) {
        return;
    }
    const ggml_cuda_kernel_launch_params params(dim3(1, 1, 1), dim3(FLINT_NEXPERT, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(flint_topk_kernel, params, ctx.topk_deferred);
    ctx.topk_deferred = {};
}

// Attention: grid (FA_GQA/HPB, n_head_kv), block of HPB*NG warps: warp w handles Q head kv_head*FA_GQA + hb*HPB + w%HPB
// and the tiles w/HPB, w/HPB + NG, ... of FA_TILE cells of each chunk of FA_CHUNK cells. K and V of a chunk are loaded into
// shared memory (the first chunk with unmasked cells before the PDL wait: the cache rows of earlier tokens are already
// written, the mask and the cell indices are graph inputs). Two lanes score one cell, the softmax state is kept per warp
// (lane holds 4 dims) and the NG states of a head are combined in shared memory. The K/V of the current token are computed
// locally and replace that cell in shared memory, block 0 also writes them to the cache. A block reads only the Q rows of
// its heads before it writes their outputs, so dst may reuse the memory of Q (see ggml_cuda_flint_attn_supported).
#define FA_D      128
#define FA_GQA    8
#define FA_TILE   16 // cells per warp and tile: lanes l and l + 16 score cell l, half of the dims each
#define FA_CHUNK  128
#define FA_KPAD   8 // K rows of 272 bytes: the 16 byte reads of 8 consecutive rows hit different banks
#define FA_MAX_KV_HEADS 8
#define FA_MAX_KV (64*FA_CHUNK)
#define FA_HPB    1 // Q heads per block
#define FA_NG     8 // warps per Q head

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

template <int rope_mode, int HPB, int NG>
static __global__ void __launch_bounds__(HPB*NG*WARP_SIZE) flint_attn_kernel(
        const float * q, const int64_t q_s1, const float * k, const int64_t k_s1, const float * v, const int64_t v_s1,
        const float * qw, const float * kw, const float eps_q, const float eps_k,
        const int32_t * pos_p, const flint_attn_rope rp,
        half * kc, half * vc, const int64_t kc_s1, const int64_t vc_s1, const int64_t * kidx, const int64_t * vidx,
        const half * mask, const int n_kv, const float * sinks, const float kq_scale,
        float * dst, block_q8_1 * yq) {
    constexpr int n_warps  = HPB*NG;
    constexpr int nthreads = n_warps*WARP_SIZE;
    constexpr int n_tiles  = FA_CHUNK/FA_TILE;
    static_assert(n_warps >= 2 && NG <= n_tiles && (HPB*8 <= WARP_SIZE || HPB*8 % WARP_SIZE == 0), "bad block shape");
    constexpr unsigned int fmask = HPB*8 >= WARP_SIZE ? 0xFFFFFFFF : (1u << (HPB*8)) - 1;

    extern __shared__ __align__(16) char fa_smem[];
    half (*Ks)[FA_D + FA_KPAD] = (half (*)[FA_D + FA_KPAD]) fa_smem;
    half (*Vs)[FA_D]           = (half (*)[FA_D]) (fa_smem + FA_CHUNK*(FA_D + FA_KPAD)*sizeof(half));
    half * mask_s              = (half *) (fa_smem + FA_CHUNK*(2*FA_D + FA_KPAD)*sizeof(half));

    __shared__ __align__(16) float qs[HPB][FA_D];
    __shared__ __align__(16) float Os[n_warps][FA_D];
    __shared__ float Ms[n_warps];
    __shared__ float Ls[n_warps];
    __shared__ __align__(16) half kcur[FA_D];
    __shared__ __align__(16) half vcur[FA_D];
    __shared__ float sinks_s[HPB];
    __shared__ unsigned long long chunk_valid; // bit per chunk of up to 64 with unmasked cells

    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int w    = tid / WARP_SIZE;
    const int g    = blockIdx.y;
    const int hl   = w % HPB;
    const int gr   = w / HPB;
    const int h0   = g*FA_GQA + blockIdx.x*HPB;

    const int n_chunks = (n_kv + FA_CHUNK - 1) / FA_CHUNK;

    // graph inputs, weights and cache rows of earlier tokens: safe to read before the wait. All loads first.
    constexpr int mask_iters = FA_MAX_KV/(8*nthreads);
    const int64_t cur_k = kidx[0];
    const int64_t cur_v = vidx[0];
    const int     pos   = rope_mode != 0 ? pos_p[0] : 0;
    const float4  qwv   = ((const float4 *) qw)[lane];
    const float4  kwv   = ((const float4 *) kw)[lane];
    const float   sinkv = sinks && tid < HPB ? sinks[h0 + tid] : -INFINITY;
    uint4 m8[mask_iters];
#pragma unroll
    for (int it = 0; it < mask_iters; ++it) {
        const int c = 8*(tid + it*nthreads); // n_kv % 8 == 0 is checked on the host
        m8[it] = c < n_kv ? *(const uint4 *) (mask + c) : make_uint4(0xFC00FC00, 0xFC00FC00, 0xFC00FC00, 0xFC00FC00);
    }

    if (tid == 0) {
        chunk_valid = 0;
    }
    if (tid < HPB) {
        sinks_s[tid] = sinkv;
    }
    __syncthreads();
    {
        unsigned long long cv = 0;
#pragma unroll
        for (int it = 0; it < mask_iters; ++it) {
            const int c = 8*(tid + it*nthreads);
            if (c < n_kv) {
                *(uint4 *) (mask_s + c) = m8[it];
            }
            const half * mh = (const half *) &m8[it];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (__half2float(mh[i]) != -INFINITY) {
                    cv |= 1ull << ((c + i)/FA_CHUNK);
                }
            }
        }
        if (cv) {
            atomicOr(&chunk_valid, cv);
        }
    }
    float rc[4];
    float rs[4];
    flint_rope_cs<rope_mode>(pos, rp, rc, rs);
    __syncthreads();
    const unsigned long long valid_chunks = chunk_valid;

    const auto load_chunk = [&](const int ch) {
#pragma unroll
        for (int r = 0; r < (2*FA_CHUNK*(FA_D/8) + nthreads - 1)/nthreads; ++r) {
            const int id = tid + r*nthreads;
            if (id >= 2*FA_CHUNK*(FA_D/8)) {
                break;
            }
            const int is_v = id / (FA_CHUNK*(FA_D/8));
            const int j    = id / (FA_D/8) % FA_CHUNK;
            const int part = id % (FA_D/8);
            const int c    = ch*FA_CHUNK + j;
            if (c < n_kv && __half2float(mask_s[c]) != -INFINITY) {
                if (is_v) {
                    cp_async_cg_16<0>(__cvta_generic_to_shared(&Vs[j][8*part]), vc + c*vc_s1 + g*FA_D + 8*part);
                } else {
                    cp_async_cg_16<0>(__cvta_generic_to_shared(&Ks[j][8*part]), kc + c*kc_s1 + g*FA_D + 8*part);
                }
            }
        }
    };

    // the chunk of the current cell always has an unmasked cell
    int ch = valid_chunks ? __ffsll(valid_chunks) - 1 : n_chunks;
    if (ch < n_chunks) {
        load_chunk(ch);
    }
    // the next kernel may start (it cannot run before the previous kernel completes)
    ggml_cuda_pdl_lc();

    ggml_cuda_pdl_sync();

    for (int r = w; r < HPB + 2; r += n_warps) {
        if (r < HPB) {
            const float4 x = flint_norm_rope<rope_mode>(((const float4 *) (q + (h0 + r)*q_s1))[lane], qwv, eps_q, rc, rs);
            ((float4 *) qs[r])[lane] = make_float4(x.x*kq_scale, x.y*kq_scale, x.z*kq_scale, x.w*kq_scale);
        } else if (r == HPB) {
            const uint2 kh = flint_f4_to_h4(flint_norm_rope<rope_mode>(((const float4 *) (k + g*k_s1))[lane], kwv, eps_k, rc, rs));
            ((uint2 *) kcur)[lane] = kh;
            if (blockIdx.x == 0) {
                *(uint2 *) (kc + cur_k*kc_s1 + g*FA_D + 4*lane) = kh;
            }
        } else {
            const uint2 vh = flint_f4_to_h4(((const float4 *) (v + g*v_s1))[lane]);
            ((uint2 *) vcur)[lane] = vh;
            if (blockIdx.x == 0) {
                *(uint2 *) (vc + cur_v*vc_s1 + g*FA_D + 4*lane) = vh;
            }
        }
    }

    float  M = -INFINITY;
    float  L = 0.0f;
    float4 O = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    while (ch < n_chunks) {
        cp_async_wait_all();
        __syncthreads();
        const int c0 = ch*FA_CHUNK;
        if (tid < FA_D/8 && cur_k >= c0 && cur_k < c0 + FA_CHUNK) {
            ((uint4 *) Ks[cur_k - c0])[tid] = ((const uint4 *) kcur)[tid];
        } else if (tid >= WARP_SIZE && tid < WARP_SIZE + FA_D/8 && cur_v >= c0 && cur_v < c0 + FA_CHUNK) {
            ((uint4 *) Vs[cur_v - c0])[tid - WARP_SIZE] = ((const uint4 *) vcur)[tid - WARP_SIZE];
        }
        __syncthreads();

#pragma unroll 1
        for (int t = gr; t < n_tiles; t += NG) {
            // lanes l and l + 16 score cell c0 + t*FA_TILE + l % 16, dims [64*(l/16), 64*(l/16) + 64)
            const int   jc = lane % FA_TILE;
            const int   c  = c0 + t*FA_TILE + jc;
            const float mv = c < n_kv ? __half2float(mask_s[c]) : -INFINITY;
            float sc = -INFINITY;
            if (mv != -INFINITY) {
                const int      dh = (lane / FA_TILE)*(FA_D/2);
                const float4 * q4 = (const float4 *) (qs[hl] + dh);
                const uint4  * k8 = (const uint4  *) (Ks[t*FA_TILE + jc] + dh);
                float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                for (int i = 0; i < FA_D/16; ++i) {
                    const uint4  kk = k8[i];
                    const float4 qa = q4[2*i + 0];
                    const float4 qb = q4[2*i + 1];
                    const float2 k0 = __half22float2(*(const half2 *) &kk.x);
                    const float2 k1 = __half22float2(*(const half2 *) &kk.y);
                    const float2 k2 = __half22float2(*(const half2 *) &kk.z);
                    const float2 k3 = __half22float2(*(const half2 *) &kk.w);
                    acc[0] += qa.x*k0.x + qa.y*k0.y;
                    acc[1] += qa.z*k1.x + qa.w*k1.y;
                    acc[2] += qb.x*k2.x + qb.y*k2.y;
                    acc[3] += qb.z*k3.x + qb.w*k3.y;
                }
                sc = (acc[0] + acc[1]) + (acc[2] + acc[3]);
            }
            // the masked lanes of both halves agree, the sum is the same on both lanes
            sc += __shfl_xor_sync(0xFFFFFFFF, sc, FA_TILE);
            sc = mv != -INFINITY ? sc + mv : -INFINITY;
            const float mt = warp_reduce_max(sc);
            if (mt == -INFINITY) {
                continue;
            }
            const float Mn   = fmaxf(M, mt);
            const float corr = expf(M - Mn);
            const float p    = expf(sc - Mn);
            // both halves hold p: half of the sum
            L = L*corr + 0.5f*warp_reduce_sum(p);
            O.x *= corr; O.y *= corr; O.z *= corr; O.w *= corr;
            M = Mn;
            // the rows of masked cells were not loaded: skip them (p == 0)
            float4 O1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
            for (int j = 0; j < FA_TILE; j += 2) {
                const float  p0 = __shfl_sync(0xFFFFFFFF, p, j + 0);
                const float  p1 = __shfl_sync(0xFFFFFFFF, p, j + 1);
                const uint2  va = ((const uint2 *) Vs[t*FA_TILE + j + 0])[lane];
                const uint2  vb = ((const uint2 *) Vs[t*FA_TILE + j + 1])[lane];
                if (p0 != 0.0f) {
                    const float2 v0 = __half22float2(*(const half2 *) &va.x);
                    const float2 v1 = __half22float2(*(const half2 *) &va.y);
                    O.x += p0*v0.x; O.y += p0*v0.y; O.z += p0*v1.x; O.w += p0*v1.y;
                }
                if (p1 != 0.0f) {
                    const float2 v0 = __half22float2(*(const half2 *) &vb.x);
                    const float2 v1 = __half22float2(*(const half2 *) &vb.y);
                    O1.x += p1*v0.x; O1.y += p1*v0.y; O1.z += p1*v1.x; O1.w += p1*v1.y;
                }
            }
            O.x += O1.x; O.y += O1.y; O.z += O1.z; O.w += O1.w;
        }

        // next chunk with unmasked cells
        const unsigned long long rest = ch + 1 < 64 ? valid_chunks >> (ch + 1) : 0;
        ch = rest ? ch + __ffsll(rest) : n_chunks;
        if (ch < n_chunks) {
            __syncthreads();
            load_chunk(ch);
        }
    }

    // combine the NG states of each head
    ((float4 *) Os[w])[lane] = O;
    if (lane == 0) {
        Ms[w] = M;
        Ls[w] = L;
    }
    __syncthreads();

    // 8 threads per head, 16 dims each
    if (tid < HPB*8) {
        const int hh = tid / 8;
        const int d0 = (tid % 8)*16;
        float Mt = sinks_s[hh];
#pragma unroll
        for (int k = 0; k < NG; ++k) {
            Mt = fmaxf(Mt, Ms[k*HPB + hh]);
        }
        float  Lt = sinks ? expf(sinks_s[hh] - Mt) : 0.0f;
        float4 o[4] = {};
#pragma unroll
        for (int k = 0; k < NG; ++k) {
            const int   ws = k*HPB + hh;
            const float f  = Ms[ws] == -INFINITY ? 0.0f : expf(Ms[ws] - Mt);
            Lt += f*Ls[ws];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 os = ((const float4 *) (Os[ws] + d0))[i];
                o[i].x += f*os.x; o[i].y += f*os.y; o[i].z += f*os.z; o[i].w += f*os.w;
            }
        }
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            o[i].x /= Lt; o[i].y /= Lt; o[i].z /= Lt; o[i].w /= Lt;
            ((float4 *) (dst + (h0 + hh)*FA_D + d0))[i] = o[i];
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(o[i].x), fabsf(o[i].y)), fmaxf(fabsf(o[i].z), fabsf(o[i].w))));
        }

        // q8_1 as quantize_q8_1: 2 threads per block
        amax = fmaxf(amax, __shfl_xor_sync(fmask, amax, 1, 8));
        const float d = amax / 127.0f;
        block_q8_1 * yb = yq + ((h0 + hh)*FA_D + d0)/QK8_1;
        int sum = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int8_t q0 = amax == 0.0f ? 0 : roundf(o[i].x / d);
            const int8_t q1 = amax == 0.0f ? 0 : roundf(o[i].y / d);
            const int8_t q2 = amax == 0.0f ? 0 : roundf(o[i].z / d);
            const int8_t q3 = amax == 0.0f ? 0 : roundf(o[i].w / d);
            sum += q0 + q1 + q2 + q3;
            *(char4 *) (yb->qs + d0 % QK8_1 + 4*i) = make_char4(q0, q1, q2, q3);
        }
        sum += __shfl_xor_sync(fmask, sum, 1, 8);
        if (tid % 2 == 0) {
            yb->ds = make_half2(d, d * sum);
        }
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

static size_t flint_attn_smem(const int n_kv) {
    return FA_CHUNK*(2*FA_D + FA_KPAD)*sizeof(half) + (size_t) n_kv*sizeof(half);
}

static bool flint_overlap(const ggml_tensor * a, const int64_t na, const ggml_tensor * b, const int64_t nb) {
    const char * a0 = (const char *) a->data;
    const char * b0 = (const char *) b->data;
    return a0 < b0 + nb && b0 < a0 + na;
}

static bool flint_attn_row_ok(const ggml_tensor * t, const int64_t nrows) {
    return t->type == GGML_TYPE_F32 && t->ne[0] == FA_D && t->ne[1] == nrows && t->ne[2] == 1 && t->ne[3] == 1 &&
        t->nb[0] == sizeof(float) && t->nb[1] % 16 == 0 && ggml_cuda_is_aligned(t, 16);
}

bool ggml_cuda_flint_attn_supported(ggml_backend_cuda_context & ctx, const ggml_cuda_flint_attn_args & a) {
    GGML_UNUSED(ctx);
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

    if (fa->type != GGML_TYPE_F32 || !ggml_is_contiguous(fa) || fa->ne[0] != FA_D || fa->ne[2] != 1 || !ggml_cuda_is_aligned(fa, 16) ||
            Q->ne[0] != FA_D || Q->ne[1] != 1 || Q->ne[2] != FA_GQA*n_head_kv || Q->ne[3] != 1 ||
            n_head_kv > FA_MAX_KV_HEADS || n_kv > FA_MAX_KV || max_bias != 0.0f || softcap != 0.0f ||
            fa->src[5] != nullptr || mask == nullptr || mask->type != GGML_TYPE_F16 || mask->ne[0] != n_kv ||
            mask->ne[2] != 1 || mask->ne[3] != 1 ||
            (sinks && (sinks->type != GGML_TYPE_F32 || sinks->ne[0] != Q->ne[2] || !ggml_is_contiguous(sinks)))) {
        return false;
    }
    // K/V: F16 cache views [D, n_kv, n_head_kv], heads adjacent within a cell
    for (const ggml_tensor * t : { K, V }) {
        if (t->type != GGML_TYPE_F16 || t->ne[0] != FA_D || t->ne[2] != n_head_kv || t->ne[3] != 1 ||
                t->nb[0] != sizeof(half) || t->nb[2] != FA_D*sizeof(half) || t->nb[1] % 16 != 0 ||
                !ggml_cuda_is_aligned(t, 16)) {
            return false;
        }
    }
    // the SET_ROWS write one row of all heads to the cells K/V read
    for (const auto & [set, kv] : { std::make_pair(a.k_set, K), std::make_pair(a.v_set, V) }) {
        if (set->type != GGML_TYPE_F16 || set->data != kv->data || set->nb[1] != kv->nb[1] ||
                set->ne[0] != FA_D*n_head_kv || set->src[1]->type != GGML_TYPE_I64 ||
                ggml_nelements(set->src[1]) != 1 || ggml_nelements(set->src[0]) != FA_D*n_head_kv) {
            return false;
        }
    }
    if (!flint_attn_row_ok(a.q, Q->ne[2]) || !flint_attn_row_ok(a.k, n_head_kv) || !flint_attn_row_ok(a.v, n_head_kv)) {
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
    const int64_t n_head = Q->ne[2];
    const int64_t q_size = (n_head - 1)*a.q->nb[1] + FA_D*sizeof(float);
    const int64_t k_size = (n_head_kv - 1)*a.k->nb[1] + FA_D*sizeof(float);
    const int64_t v_size = (n_head_kv - 1)*a.v->nb[1] + FA_D*sizeof(float);
    const int64_t d_size = ggml_nbytes(fa);
    if (flint_overlap(fa, d_size, a.k, k_size) || flint_overlap(fa, d_size, a.v, v_size) ||
            (flint_overlap(fa, d_size, a.q, q_size) && (fa->data != a.q->data || a.q->nb[1] != FA_D*sizeof(float)))) {
        return false;
    }
    return n_kv % 8 == 0 && ggml_cuda_is_aligned(mask, 16) && n_kv <= FA_MAX_KV;
}

void ggml_cuda_flint_attn(ggml_backend_cuda_context & ctx, const ggml_cuda_flint_attn_args & a) {
    const ggml_tensor * fa = a.fa;
    const ggml_tensor * K  = fa->src[1];
    const ggml_tensor * V  = fa->src[2];
    const int n_head_kv = K->ne[2];
    const int n_kv      = K->ne[1];

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
    const size_t smem = flint_attn_smem(n_kv);
    const ggml_cuda_kernel_launch_params params(dim3(FA_GQA/FA_HPB, n_head_kv, 1), dim3(FA_HPB*FA_NG*WARP_SIZE, 1, 1), smem, ctx.stream());
    const auto launch = [&](auto kernel) {
        static bool smem_set[GGML_CUDA_MAX_DEVICES][3] = {};
        if (!smem_set[ctx.device][mode]) {
            CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, flint_attn_smem(FA_MAX_KV)));
            smem_set[ctx.device][mode] = true;
        }
        ggml_cuda_kernel_launch(kernel, params,
            (const float *) a.q->data, (int64_t) (a.q->nb[1]/sizeof(float)),
            (const float *) a.k->data, (int64_t) (a.k->nb[1]/sizeof(float)),
            (const float *) a.v->data, (int64_t) (a.v->nb[1]/sizeof(float)),
            (const float *) a.q_norm_w->data, (const float *) a.k_norm_w->data, a.eps_q, a.eps_k,
            pos, rp,
            (half *) K->data, (half *) V->data, (int64_t) (K->nb[1]/sizeof(half)), (int64_t) (V->nb[1]/sizeof(half)),
            (const int64_t *) a.k_set->src[1]->data, (const int64_t *) a.v_set->src[1]->data,
            (const half *) fa->src[3]->data, n_kv, fa->src[4] ? (const float *) fa->src[4]->data : nullptr, kq_scale,
            (float *) fa->data, yq);
    };
    switch (mode) {
        case 0: launch(flint_attn_kernel<0, FA_HPB, FA_NG>); break;
        case 1: launch(flint_attn_kernel<1, FA_HPB, FA_NG>); break;
        default: launch(flint_attn_kernel<2, FA_HPB, FA_NG>); break;
    }
    if (a.fa_2d) {
        ctx.q8_1_set(a.fa_2d);
    }
}
