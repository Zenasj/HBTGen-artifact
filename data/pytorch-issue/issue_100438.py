import torch
import torch.nn as nn

args = (torch.rand((4, 4, 32)),)

# Model which uses repeat_interleave
class RepeatInterleaveModel(nn.Module):
    def forward(self, x):
        return x.repeat_interleave(2, dim=-1)

model = RepeatInterleaveModel()

# Resulting onnx model has 463 nodes!
torch.onnx.export(model, args, "repeat_interleave.onnx", opset_version=17)

torch.onnx.export(model, args, "dynamic_repeat_interleave.onnx", opset_version=17,
                  input_names=['input'], output_names=['output'], dynamic_axes={'input': [2]})

# Model which produces the same result but with fewer nodes.
class UnsqueezeRepeat(nn.Module):
    def forward(self, x):
        repeats = tuple((1,) * len(x.shape) + (2,))
        return x.unsqueeze(-1).repeat(repeats).flatten(-2, -1)

alt_model = UnsqueezeRepeat()
assert torch.equal(model(*args), alt_model(*args)) # Check correctness

# Exports to only 15 nodes. Exported graph is the same for both static and dynamic versions.
torch.onnx.export(alt_model, args, "alt_repeat_interleave.onnx", opset_version=17)
torch.onnx.export(alt_model, args, "alt_dynamic_repeat_interleave.onnx", opset_version=17,
                  input_names=['input'], output_names=['output'], dynamic_axes={'input': [2]})