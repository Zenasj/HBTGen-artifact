import torch.nn as nn

# Model + export code
import torch

class RepeatInterleaveModel(torch.nn.Module):
    def forward(self, x):
        return x.repeat_interleave(2, dim=-1)

args = (torch.rand((2, 2, 16)),)
model = RepeatInterleaveModel()
torch.onnx.export(model, args, "repeat_interleave.onnx", opset_version=17)