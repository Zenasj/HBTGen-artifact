import torch.nn as nn

import torch
import onnxruntime

class Model(torch.nn.Module):
  def forward(self, x):
    aa = torch.tensor([[0],[1],[2]])
    return aa.expand_as(x)

x = torch.ones(3,2)
model = Model()
print(model(x))
model_path='./model.onnx.pb'
torch.onnx.export(model,(x,),model_path, input_names=["x"],dynamic_axes={"x": [0, 1]})


ort_session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

ort_inputs = { 'x': x.numpy() }
output_columns = []
ort_outs = ort_session.run(output_columns, ort_inputs)