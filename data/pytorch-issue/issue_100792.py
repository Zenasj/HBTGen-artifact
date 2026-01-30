import torch.nn as nn

import torch
import torch._dynamo
import torch._inductor.config as config
dropout = torch.nn.Dropout(p=0.1, inplace=False)

@config.patch(cpp_wrapper=True, lowmem_dropout=False)
@torch._dynamo.optimize("inductor")
def fn(a):
    return dropout(a)

x = torch.rand([64, 4, 128, 128])
out = fn(x)