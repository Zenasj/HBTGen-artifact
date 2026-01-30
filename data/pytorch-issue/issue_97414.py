import torch
import torch.nn as nn

def forward(self):
        # File: /data/users/ezyang/c/pytorch/test/test_torch.py:7602, code: x = torch.empty(4, 3, 8, 8, device='meta')
        empty: f32[4, 3, 8, 8] = torch.ops.aten.empty.memory_format([4, 3, 8, 8], device = device(type='meta'), pin_memory = False)
        
        # File: /data/users/ezyang/c/pytorch/test/test_torch.py:7604, code: torch._C._nn.upsample_nearest2d(x, (16, 16), out=out)
        upsample_nearest2d: f32[4, 3, 16, 16] = torch.ops.aten.upsample_nearest2d.default(empty, [16, 16])
        return (empty, upsample_nearest2d)