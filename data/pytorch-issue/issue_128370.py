import torch

def run_gemm() -> Tensor:
        x_fp8: Tensor
        w_fp8: Tensor
        x_scale: Tensor
        w_scale: Tensor
        x_fp8, x_scale = quantize_fp8_row(x)
        w_fp8, w_scale = quantize_fp8_row(w)
        return matmul_fp8_row(
            x_fp8,
            w_fp8,
            x_scale,
            w_scale,
            dot_out_dtype=torch.float32,
            allow_tf32=True,
            fp8_fast_accum=True,
        )