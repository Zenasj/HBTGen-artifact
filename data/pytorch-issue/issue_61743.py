import torch.nn as nn

import torch
import pointnet2_ops

class DummyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return pointnet2_ops.pointnet2_utils.ball_query(3.4, 5, x, x)
    
model = torch.nn.Sequential(DummyModule()).to('cuda')
model = torch.jit.script(model)