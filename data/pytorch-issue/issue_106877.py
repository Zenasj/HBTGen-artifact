import torch.nn as nn

import torch


class Float8Module(torch.nn.Module):
    def forward(self, input: torch.Tensor):
        input = input.to(torch.float8_e5m2)
        return input + torch.tensor(1.0, dtype=torch.float8_e5m2)

torch.onnx.dynamo_export(Float8Module(), torch.randn(1, 3, 224, 224))