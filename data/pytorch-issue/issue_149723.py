import torch.nn as nn

import torch
import torch.onnx
import torch.jit
from torch import nn, Tensor
import io

from torch.onnx.verification import find_mismatch

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = nn.Linear(8, 4)
        self.module2 = nn.Linear(4, 2)
    
    def forward(self, x: Tensor) -> Tensor:
        preout = self.module(x)
        out = self.module2(preout)
        return out

model = Model()
scripted_model = torch.jit.script(model)
dummy_input = torch.randn(3, 8)
opset_version = 9
graph_info = find_mismatch(model=scripted_model, 
                            input_args=(dummy_input,),
                            opset_version=opset_version,
                            verbose=False)