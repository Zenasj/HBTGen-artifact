import torch 
import torch.nn as nn

import torch._inductor.config as config 

config.force_disable_caches = True
config.max_autotune = True
config.max_autotune_gemm_backends = "CUTLASS"
# the following is only needed if you use a custom cutlass library 
# config.cuda.cutlass_dir = "/data/users/henrylhtsang/cutlass"

class TestModule(nn.Module):
    def forward(self, A, B):
        return A @ B 

model = TestModule().cuda()
M, K, N = 2048, 2048, 2048
A = torch.randn(M, K).cuda().half()
B = torch.randn(K, N).cuda().half()

C = torch.compile(model, fullgraph=True)(A, B)