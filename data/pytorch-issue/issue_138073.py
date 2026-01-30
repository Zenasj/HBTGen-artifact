import torch.nn as nn

import torch
import torch._dynamo
import torch.export

class FullConstNDynamicV(torch.nn.Module):
    def forward(self, x):
        n = 7
        v = x[0, 0]
        out = torch.full((n,), v)
        # Replacing the above line with the following will fix export 'Pending unbacked symbols' error:
        # out = torch.ones((n,)) * v

        return out

input_tensor = torch.ones(1, 100)
torch.export.export(FullConstNDynamicV(), (input_tensor, ))