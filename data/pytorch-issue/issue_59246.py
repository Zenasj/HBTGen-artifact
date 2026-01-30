import torch.nn as nn

import torch
from torch.onnx import register_custom_op_symbolic

class Model(torch.nn.Module):

    def forward(self, x):
        return torch.fft.fft(x)

def fft(g, self, n, dim, norm):
    return g.op("com.microsoft.experimental::DFT", self)
register_custom_op_symbolic("::fft_fft", fft, 1)


model = Model()
ts_model = torch.jit.script(model)

data = torch.randn(1, 1024)
y = ts_model(data)

torch.onnx.export(
    ts_model,
    (data,),
    "tmp.onnx",
    opset_version=13,
    verbose=True,
    example_outputs=(y,),
)