import torch
import torch.nn as nn


class Model(nn.Module):
  def forward(self, x):
    x = x[:, 0]
    return x


def main():
  model = Model()
  dummy_input = torch.zeros(1, 2)
  torch.onnx.export(model, dummy_input, 'model.onnx', verbose=True)


if __name__ == '__main__':
  main()

import torch
import torch.nn as nn


class Model(nn.Module):
  def forward(self, x):
    x = torch.narrow(x, dim=1, start=1, length=1)
    return x


def main():
  model = Model()
  dummy_input = torch.zeros(1, 2)
  torch.onnx.export(model, dummy_input, 'model.onnx', verbose=True)


if __name__ == '__main__':
  main()