import torch.nn as nn

import torch

inp = torch.nn.Parameter(torch.randn([8, 1, 4, 1], dtype=torch.float32), requires_grad=True)


def fn():

    v5_0 = torch.nn.functional.interpolate(inp, size=[36, 1], scale_factor=None, mode='bicubic',
                                           align_corners=None, recompute_scale_factor=None, antialias=False)
    v3_0 = torch.neg(v5_0)
    linear_layer = torch.nn.Linear(in_features=1, out_features=25, bias=True)
    v2_0 = linear_layer(v3_0)
    v1_0 = v2_0.to(torch.float64)
    tan = torch.tan(v1_0)
    return (tan,)

ret_eager = fn()
compiled = torch.compile(fn)
ret_compiled = compiled()

torch.testing.assert_close(ret_eager[0], ret_compiled[0])

# AssertionError: Tensor-likes are not close!
# 
# Mismatched elements: 7200 / 7200 (100.0%)
# Greatest absolute difference: 4132.387735664385 at index (7, 0, 14, 5) (up to 1e-07 allowed)
# Greatest relative difference: 28269.87316096882 at index (7, 0, 31, 20) (up to 1e-07 allowed)