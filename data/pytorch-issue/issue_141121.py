import torch
from torch._inductor.utils import run_and_get_triton_code
from torch._inductor import config

import functools

config.triton.max_tiles = 3
config.triton.prefer_nd_tiling = True

full_size, view_size, num_block_pointers, num_tiles = (
    (5, 5, 5, 5, 5),
    (3, 3, 5, 3, 5),
    1,
    2,
)

GPU_TYPE = "cuda"


def get_input() -> torch.Tensor:
    device = torch.device(GPU_TYPE)
    full = torch.randn(full_size).to(device)
    return torch.as_strided(full, view_size, full.stride())


a, b = get_input(), get_input()

opt_fn = torch.compile(functools.partial(torch.add))
code = run_and_get_triton_code(opt_fn, a, b)