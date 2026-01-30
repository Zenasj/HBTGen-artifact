import torch.nn as nn

import os
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"] = "ATEN,TRITON,CPP,CUTLASS"
# os.environ["TORCH_LOGS"] = "+dynamo"
import logging

import torch
import torch._inductor.config as config
import torch._inductor.codegen.cuda.gemm_template as gemm_template

gemm_template.log.setLevel(logging.INFO)
config.cuda.cutlass_dir = "../../cutlass"
config.debug = True

x = torch.randn(8192, 2048, dtype=torch.bfloat16, device="cuda")
y = torch.randn(2048, 2048, dtype=torch.bfloat16, device="cuda")

def gemm(a, b):
    return torch.nn.functional.linear(a, b)

compiled_gemm = torch.compile(gemm, mode="max-autotune")
z = compiled_gemm(x, y)