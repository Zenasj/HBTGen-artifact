import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
  def forward(self, x):
    x = F.avg_pool2d(x, 2)
    return x


def main():
  model = Model()
  dummy_input = torch.zeros(1, 1, 4, 4)
  torch.onnx.export(model, dummy_input, 'model.onnx', verbose=True)


if __name__ == '__main__':
  main()

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
  def forward(self, x):
    x = F.max_pool2d(x, 2)
    return x


def main():
  model = Model()
  dummy_input = torch.zeros(1, 1, 4, 4)
  torch.onnx.export(model, dummy_input, 'model.onnx', verbose=True)


if __name__ == '__main__':
  main()