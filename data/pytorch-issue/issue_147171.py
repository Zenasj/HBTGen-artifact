import random

import torch

@torch.compile()
def foo(y):
    x = torch.rand([10])
    return y + 2

foo(torch.rand([4], device="cuda"))

def forward(self, arg0_1: "f32[4][1]cuda:0"):
        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[1][1]cpu" = torch.ops.prims.inductor_seeds.default(1, device(type='cpu'))
        inductor_lookup_seed_default: "i64[][]cpu" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[10][1]cpu" = torch.ops.prims.inductor_random.default([10], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = inductor_random_default = None
        
         # File: /data/users/eellison/pytorch/work_dir/test_hi5.py:7 in foo, code: return y + 2
        add: "f32[4][1]cuda:0" = torch.ops.aten.add.Tensor(arg0_1, 2);  arg0_1 = None
        return (add,)