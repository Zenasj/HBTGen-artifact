import torch
import torch._inductor.config as inductor_config
import triton

M = 20120
K = 512
N = 1536

a = torch.randn([M,N]).cuda()
b = torch.randn([M,K]).cuda()
c = torch.randn([K,N]).cuda()

def mm():
    return torch.addmm(a, b, c)

with inductor_config.patch(
    max_autotune=True,
    max_autotune_gemm_backends="TRITON",
    autotune_fallback_to_aten=False,
):
    pt2_mm = torch.compile(mm, dynamic=False)
    pt2_mm()

if __name__ == "__main__":
    pt2_mm()