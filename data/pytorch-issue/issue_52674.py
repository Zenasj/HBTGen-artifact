import torch.nn as nn

import torch
from typing import Dict

class M(torch.nn.Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(M, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, d: Dict[str, torch.Tensor]) -> torch.Tensor:
        return d["feature"]


sc = torch.jit.script(M(1, 1, 0.125))
torch.jit.save(sc, 'scriptmodule.pt')
loaded = torch.jit.load("scriptmodule.pt")
d = dict(feature=torch.rand(2, 3))
loaded(d)