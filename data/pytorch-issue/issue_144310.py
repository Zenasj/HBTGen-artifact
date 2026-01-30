import torch.nn as nn

import torch
torch._dynamo.config.recompile_limit = 12

def run_test(dim, dtype):
    x = torch.randn([2] * (dim + 2)).to(dtype)
    op = eval(f"torch.nn.functional.avg_pool{dim}d")
    try:
        op(x, kernel_size=2, stride=2)
        print("succeed on eager")
    except Exception as e:
        print(e)

    try:
        torch.compile(op)(x, kernel_size=2, stride=2)
        print("succeed on inductor")
    except Exception as e:
        print(e)


for dim in (1, 2, 3):
    for dtype in (torch.uint8, torch.uint16, torch.uint32, torch.uint64):
        run_test(dim, dtype)