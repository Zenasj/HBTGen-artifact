torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)

# flash_attn, autocast, non-compiled 1.0096447467803955, mem 137372160
# math_sdp, autocast, compiled 0.7690215110778809 mem 291901952
# math, autocast, non-compiled 1.101301908493042, mem 174375936

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

p = model()
p = torch.compile(p)

with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    p(bev, positions, final_pos)

torch.cuda.synchronize()
torch.cuda.empty_cache()
warmed_up = torch.cuda.memory_allocated()
print("warmed up", warmed_up)

start = time.time()
for i in range(200):
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        out = p(bev, positions, final_pos)
torch.cuda.synchronize()
print(time.time()-start)
print("final", torch.cuda.memory_allocated())
print("delta", torch.cuda.memory_allocated()-warmed_up)
assert out[0].requires_grad

import torch
import time
from torchdrive.models.path import PathTransformer

device = torch.device('cuda')

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


def run():
    base = torch.cuda.memory_allocated()
    print("base", base)
    
    BS = 8
    p = PathTransformer(
        bev_shape=(16, 16),
        bev_dim=384,
        dim=384,
    ).to(device)
    p = torch.compile(p)
    bev = torch.rand(BS, 384, 16, 16, device=device)
    positions = torch.rand(BS, 3, 120, device=device)
    final_pos = torch.rand(BS, 3, device=device)

    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        p(bev, positions, final_pos)

    # flash_attn, autocast, non-compiled 1.0096447467803955, mem 137372160
    # math_sdp, autocast, compiled 0.7690215110778809 mem 291901952
    # math, autocast, non-compiled 1.101301908493042, mem 174375936

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    warmed_up = torch.cuda.memory_allocated()
    print("warmed up", warmed_up)
    
    start = time.time()
    for i in range(200):
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = p(bev, positions, final_pos)
    torch.cuda.synchronize()
    print(time.time()-start)
    print("final", torch.cuda.memory_allocated())
    print("delta", torch.cuda.memory_allocated()-warmed_up)
    print(out[0].dtype)
    assert out[0].requires_grad
run()