import torch.nn as nn

import torch


# Minified repro from bench/huggingface MobileBertForMaskedLM
# RuntimeErrorWithDiagnostic: FX Node: call_function:aten.cat.default[name=cat]. Raised from:
# KeyError: 'val'
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(3))

    def forward(self, x):
        return torch.cat([x, self.param])

model = Model()
torch.onnx.dynamo_export(model, torch.randn(3))