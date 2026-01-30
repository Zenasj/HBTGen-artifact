import torch.nn as nn

import torch

class BroadcastInDim(torch.nn.Module):
    def forward(self, x):
        x = x.clone()
        dims = [0, 1]
        shape = [x.shape[d] for d in dims]
        shape.append(1)
        return torch.ops.prims.broadcast_in_dim.default(x, shape, dims).clone()
    
x = torch.randn(3, 4)
dim = torch.export.Dim("dyn_dim", min=1, max=4)
mod = torch.export.export(BroadcastInDim(), (x,), dynamic_shapes=({0: dim},))