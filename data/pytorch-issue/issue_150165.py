import torch.nn as nn

import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, size=(64, 64), mode='bilinear', align_corners=True)
        x = torch.inverse(x)
        return x

model = Model()
inputs = torch.randn(1, 3, 32, 32)
res = model(inputs)

compiled_model = torch.compile(model, backend='inductor')
with torch.no_grad():
    compiled_out = compiled_model(inputs)
torch.testing.assert_close(res, compiled_out)