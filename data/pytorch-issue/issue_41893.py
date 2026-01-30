import torch
import torch.nn as nn

class BadOne(nn.Module):
  def forward(self, x):
    return x.flatten(2, 4)

torch.onnx.export(badModel, input, "badModel.onnx",
                 input_names = ['input'],
                 output_names = ['output'],
                 verbose=True,
                 keep_initializers_as_inputs=True,
                 dynamic_axes = {'input' : {1 : 'seqLen'},'output' : {1 : 'seqLen'}})

class GoodOne(nn.Module):
  def forward(self, x):
    return x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3] * x.shape[4]))