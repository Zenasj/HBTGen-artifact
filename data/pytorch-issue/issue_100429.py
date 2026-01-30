import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        return x.repeat_interleave(2)

model = MyModel()
args = (torch.ones(1, 3),)

print("model.forward() succeeeds: ", model(*args).shape)

torch.onnx.export(
    model,
    args,
    "repeat_interleave.onnx",
    input_names=["input"], 
    output_names=["output"], 
    dynamic_axes={"input": [0]},
    opset_version=17,
)