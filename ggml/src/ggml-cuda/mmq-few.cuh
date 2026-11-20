#pragma once

#include "common.cuh"

bool ggml_cuda_should_use_mmq_few(const ggml_tensor * src0, const ggml_tensor * src1, int cc);

// res: optional residual added to the result (a fused ADD into dst).
// src1_q8_1: optional q8_1 rows of src1 (ggml_backend_cuda_context::q8_1_lookup) read instead of quantizing src1, for a src1
// that has no F32 data; quantizing again to the fragment layout is faster than gathering the fragments from q8_1 rows.
void ggml_cuda_mul_mat_q_few(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_tensor * res = nullptr, const void * src1_q8_1 = nullptr);

// src1_q8_1 can be used: one channel of rows without padding, a src0 type that needs no src1 sums
bool ggml_cuda_mul_mat_q_few_q8_1_ok(const ggml_tensor * src0, const ggml_tensor * src1);
