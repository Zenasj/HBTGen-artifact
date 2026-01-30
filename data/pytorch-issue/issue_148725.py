import torch
import torch._inductor.config as config

config.triton.prefer_nd_tiling = True
config.triton.use_block_ptr = True

full_size = (114, 10, 160)
def get_input(view_size):
    full = torch.rand(full_size, device="cuda")
    view = torch.as_strided(full, view_size, full.stride())
    return view

inps = [
    get_input(view_size)
    for view_size in [(114, 10, 140), (114, 1, 140)]
]

compiled = torch.compile(torch.add)
compiled(*inps)