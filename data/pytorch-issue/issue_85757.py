import torch.nn as nn

import torch
from typing import Dict

class Module(torch.nn.Module):
    def forward(
        self, x: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return x

model = torch.jit.script(Module())
x = {"input": torch.zeros(1)}
torch.onnx.export(model, (x, {}), "out", opset_version=9)  # ERR