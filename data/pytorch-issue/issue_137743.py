import torch.nn as nn
import torchvision

import torch
from torchvision.transforms.v2.functional import resize

import os

class Resizer(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return resize(x, size=(512,512))
    
class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.resizer = Resizer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resizer.forward(x.permute(0, 1, 2))

with torch.no_grad():
    device="cuda:0"
    model = Model().to(device=device)
    inp=torch.randn((1024, 1024, 3), device=device)

    so_path = torch._export.aot_compile(
        model,
        (inp,),
        options={"aot_inductor.output_path": os.path.join("/tmp", "model.so")},
    )

    torch._export.aot_load(
        so_path,
        device=device
    )