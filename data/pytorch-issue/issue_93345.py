import torch.nn as nn

import torch.onnx
import torch

class Model(torch.nn.Module):
    def forward(self, x):
        y = torch.quantize_per_tensor(x, 1., 0, torch.qint8)
        return y.int_repr()

model = Model()
model.eval()
args = torch.ones(1, 1, 28, 28)
print(model(args))

torch.onnx.export(model,               # model being run
                  args,                         # model input (or a tuple for multiple inputs)
                  "out.onnx",   # where to save the model (can be a file or file-like object)
                  opset_version=16,          # the ONNX version to export the model to
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'])