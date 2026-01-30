import random

import torch
import torch.nn as nn
import numpy as np
import io

class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.qconfig = torch.quantization.default_qconfig
        self.conv = torch.quantization.QuantWrapper(
            nn.Conv2d(3, 5, 2, bias=True).to(dtype=torch.float)
        )
    def forward(self, x):
        return self.conv(x)

torch.backends.quantized.engine = 'qnnpack'
opset_version = 10

model = ConvModel()
model = torch.quantization.prepare(model)
model = torch.quantization.convert(model)

x_numpy = np.random.rand(1, 3, 6, 6).astype(np.float32)
x = torch.from_numpy(x_numpy).to(dtype=torch.float)
outputs = model(x)
input_names = ["x"]

traced = torch.jit.trace(model, x)
buf = io.BytesIO()
torch.jit.save(traced, buf)
buf.seek(0)

model = torch.jit.load(buf)
f = io.BytesIO()
torch.onnx.export(model, x, f, input_names=input_names, example_outputs=outputs,
                    operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                    opset_version=opset_version
)
f.seek(0)