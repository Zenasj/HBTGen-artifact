import torch.nn as nn

py
import logging
import os

os.environ["TORCH_LOGS"] = "+output_code,+benchmarking,+inductor"

import torch

import torch._inductor.config

torch._inductor.config.max_autotune = True
# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.coordinate_descent_check_all_directions = True
torch._inductor.config.force_disable_caches = True
torch._inductor.config.autotune_num_choices_displayed = None
# torch._inductor.config.autotune_in_subproc = False
torch._inductor.config.max_autotune_gemm_backends = "CUTLASS,TRITON"
torch._inductor.config.cuda.cutlass_max_profiling_configs = 2
torch._inductor.config.cuda.cutlass_instantiation_level = "3333"
torch._inductor.config.cuda.cutlass_op_allowlist_regex = "cutlass3x_sm90_tensorop_s64x56x16gemm_f16_f16_f32_void_f16_128x112x64_4x1x1_0_ttn_align8_warpspecialized_pingpong_epi_tma"

class MatMulModel(torch.nn.Module):
    def forward(self, A, B):
        return A @ B


def main():
    M, N, K = 2048, 2048, 2048
    dtype = torch.float16
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)
    model = MatMulModel().cuda()

    compiled_model = torch.compile(model, fullgraph=True)
    _ = compiled_model(A, B)

    print("done")


if __name__ == "__main__":
    main()