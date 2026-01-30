import os
import torch
import torch.nn as nn

from torch._inductor import config as inductor_config
from torch._inductor.utils import fresh_inductor_cache

M, N, K = 128, 128, 4096
dtype = torch.float16

X = torch.randn(M, N, dtype=dtype).cuda()
A = torch.randn(M, K, dtype=dtype).cuda()
B = torch.randn(K, N, dtype=dtype).cuda()


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, b, x, y):
        return torch.addmm(b, x, y)


import ck4inductor
ck_dir = os.path.dirname(ck4inductor.__file__)

with fresh_inductor_cache():
    with inductor_config.patch(
        {
            "max_autotune_gemm_backends": "CK",
            "autotune_fallback_to_aten": False,
            "compile_threads": 144,
            "rocm.ck_dir": ck_dir,
        }
    ):
        compiled_model = torch.compile(SimpleModel(), mode="max-autotune")
        res = compiled_model(X, A, B)
        res_eager = torch.addmm(X, A, B)
        torch.testing.assert_close(res, res_eager)