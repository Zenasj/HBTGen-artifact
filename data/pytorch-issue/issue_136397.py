import torch

A = rand_sparse_semi_structured_mask(256, 128, dtype=torch.float16)
B = torch.rand(dense_input_shape, device=device).to(torch.float16).t()

A_fp8, A_scale = to_float8(A)
B_fp8, B_scale = to_float8(B)

dense_result = torch._scaled_mm(
    A_fp8, B_fp8,
    scale_a=A_scale, scale_b=B_scale,
    out_dtype=out_dtype
)
A_fp8_sparse = to_sparse_semi_structured(A_fp8)
sparse_result = torch._scaled_mm(
    A_fp8_sparse, B_fp8,
    scale_a=A_scale, scale_b=B_scale,
    out_dtype=out_dtype
)