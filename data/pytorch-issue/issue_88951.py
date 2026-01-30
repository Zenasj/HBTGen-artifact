import torch.nn as nn

import torch

def test():
    arg_1 = torch.rand([5, 256, 16, 16], dtype=torch.float32).clone()
    res = torch.nn.functional.interpolate(arg_1,None,-1e20,"bilinear",False,)

test()