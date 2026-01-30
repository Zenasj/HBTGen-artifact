import torch
import torch.nn as nn

class MLPModel(nn.Module):
  def __init__(self):
      super().__init__()
      self.lstm = nn.LSTM(960, 480, bias=True)

  def forward(self, tensor_x: torch.Tensor):
      x, h = self.lstm(tensor_x)
      return x

model = MLPModel()
tensor_x = torch.rand((1, 960), dtype=torch.float32)
print(model(tensor_x).shape)
onnx_program = torch.onnx.dynamo_export(model, tensor_x).save('model.onnx')

import torch
import torch.nn as nn

class MLPModel(nn.Module):
  def __init__(self):
      super().__init__()
      self.lstm = nn.LSTMCell(960, 480, bias=True)

  def forward(self, tensor_x: torch.Tensor):
      x, h = self.lstm(tensor_x)
      return x

model = MLPModel()
tensor_x = torch.rand((1, 960), dtype=torch.float32)
print(model(tensor_x).shape)
onnx_program = torch.onnx.dynamo_export(model, tensor_x).save('model.onnx')