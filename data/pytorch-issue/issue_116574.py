import torch.nn as nn

import torch
from torch import nn
class ONNXBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(10, bias=False) # <------ HERE
    def forward(self, x):
        return self.ln(x)
model = ONNXBug()
dummy = torch.zeros(10).unsqueeze(0)
torch.onnx.export(model, dummy, "test.onnx", export_params=True, input_names = ['input'], output_names = ['output'])
import onnxruntime as rt
sess = rt.InferenceSession("./test.onnx")