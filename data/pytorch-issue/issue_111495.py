import torch.nn as nn

import torch

class LinearModel(torch.nn.Module):
  def __init__(self):
      super().__init__()
      self.fc0 = torch.nn.Linear(5, 7)

  def forward(self, tensor_x: torch.Tensor):
      output = self.fc0(tensor_x)
      return output

def linearDataloader():
    yield torch.randn(3, 5).cuda()

# Get model and input
model = LinearModel()
data = next(linearDataloader())

# ONNX Export
export_output = torch.onnx.dynamo_export(
    model.eval().to('cuda'),
    data
)
export_output.save('linear_dynamo.onnx')

import torch

class LinearModel(torch.nn.Module):
  def __init__(self):
      super().__init__()
      self.fc0 = torch.nn.Linear(5, 7)

  def forward(self, tensor_x: torch.Tensor):
      output = self.fc0(tensor_x)
      return output

def linearDataloader():
    yield torch.randn(3, 5).cuda()

# Get model and input
model = LinearModel()
data = next(linearDataloader())

# ONNX Export
export_output = torch.onnx.dynamo_export(
    model.eval().to('cuda'),
    data
)

import onnx
flat_graph = onnx.inliner.inline_local_functions(export_output.model_proto)