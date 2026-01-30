import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, *, inp):
        # Must be called with inp=...
        return inp


model = MyModel()

print("Forward pass works: ", model(inp=torch.zeros(4)), "\n")

# Keyword arguments given to torch.onnx.export as a dictionary
# See https://pytorch.org/docs/stable/onnx.html#torch.onnx.export
args = ({"inp": torch.zeros(4)},)

# Fails
torch.onnx.export(model, args, "pytorch_test.onnx")

torch.onnx.export(model, (torch.zeros(4),), "pytorch_test.onnx")