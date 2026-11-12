#pragma once

#include "common.cuh"

// GGML_OP_MUL_MAT_ID_MULTI: mul_mat_id over experts split into tensors of different types
bool ggml_cuda_mul_mat_id_multi_supported(const ggml_tensor * op);

// res/add_dst: fuse a following residual add (add_dst = dst + res) into a weighted op; dst itself is then not written
void ggml_cuda_mul_mat_id_multi(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * res = nullptr,
        ggml_tensor * add_dst = nullptr);

// whether the weighted op `node` can write add_dst = node + res directly (see ggml_cuda_mul_mat_id_multi)
bool ggml_cuda_mul_mat_id_multi_can_fuse_add(const ggml_tensor * node, const ggml_tensor * res, const ggml_tensor * add_dst);
