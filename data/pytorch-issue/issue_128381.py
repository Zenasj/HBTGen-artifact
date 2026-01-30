import torch
import torch._inductor.config

torch._inductor.config.force_mixed_mm = True

def f(a, b):
    return torch.mm(a, b.to(a.dtype))

fp16_act = torch.randn(1, 32).to(torch.float16).cuda()
fp8_weight = torch.randn(32, 32).to(torch.float8_e5m2).cuda()
torch.compile(f)(fp16_act, fp8_weight)

import torch
import torch._inductor.config

#torch._inductor.config.force_mixed_mm = True
torch._inductor.config.mixed_mm_choice = "triton"

def f(a, b):
    return torch.mm(a, b.to(a.dtype))

fp16_act = torch.randn(1, 32).to(torch.float16).cuda()
fp8_weight = torch.randn(32, 32).to(torch.float8_e5m2).cuda()
torch.compile(f)(fp16_act, fp8_weight)