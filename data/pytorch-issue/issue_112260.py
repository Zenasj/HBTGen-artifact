import torch

weights = torch.randn(10, 10, device="cuda")

def mod(x):
    return torch.mm(x, weights)

opt_inductor = torch.compile(mod, backend="inductor")
opt_eager = torch.compile(mod, backend="eager")
x = torch.randn(1, 10, device="cuda")

with torch.autocast('cuda', torch.float16):
    ref = mod(x)
    res = opt_inductor(x)
    res_eager = opt_eager(x)

print(ref.dtype, res.dtype, res_eager.dtype)  # float16, float16, float16

with torch.autocast('cuda', torch.bfloat16):  # No recompilation
    ref = mod(x)
    res = opt_inductor(x)
    res_eager = opt_eager(x)

print(ref.dtype, res.dtype, res_eager.dtype)  # bfloat16, float16, bfloat16

with torch.no_grad():
    ref = mod(x)
    res = opt_inductor(x)
    res_eager = opt_eager(x)

print(ref.requires_grad, res.requires_grad, res_eager.requires_grad)   # False, False, False

with torch.enable_grad():
    ref = mod(x)
    # 'Recompiling function mod in /home/jonch/Desktop/sdpa.py:1380', 'triggered by the following guard failure(s): ___check_global_state()
    res = opt_inductor(x)
    res_eager = opt_eager(x) 

print(ref.requires_grad, res.requires_grad, res_eager.requires_grad)  # True, True, True