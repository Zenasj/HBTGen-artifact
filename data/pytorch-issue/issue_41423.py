import torch
import torch.nn as nn
import numpy as np

class Demo(nn.Module):
    def __init__(self):
        super().__init__()
        self.fold = nn.Fold(output_size=(4,5), kernel_size=(2,2))
        
    def forward(self, x):
        folded = self.fold(x)
        return folded
    
input_tensor = torch.randn((1, 3*2*2, 12))
demo = Demo()
out = demo(input_tensor)

torch.onnx.export(
    demo,
    input_tensor,
    "debug.onnx",
    verbose=True,
    input_names=["data"],
    opset_version=12,
    dynamic_axes={"data": {0: "batch"}}
)