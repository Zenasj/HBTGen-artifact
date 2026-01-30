import torch.nn as nn

py
import torch

linear_model = torch.nn.Linear(1, 1, bias=False)
linear_model.weight.data = torch.tensor([7]).float()

linear_model(torch.tensor([1]).float())