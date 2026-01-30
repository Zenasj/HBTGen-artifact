import torch
import torch.nn.functional as F

@torch._dynamo.optimize("eager")
def foo(inp, w):
    return F.conv2d(inp, w)

inp = torch.rand((1, 1, 32, 32))
w = torch.rand((1, 2, 3, 3))
#                  |
#                  |--------- incorrect shape!

foo(inp, w)