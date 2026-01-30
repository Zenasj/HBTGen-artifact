import torch.nn as nn

import torch
import torch.nn.functional as F
inputs = torch.Tensor([[0, 1], [1.0, 0], [0, 1]]).to("mps")
targets = torch.LongTensor([1, 0, -100]).to("mps")
F.cross_entropy(inputs, targets, ignore_index=-1, reduction="none")

import torch
import torch.nn.functional as F
inputs = torch.Tensor([[0, 1], [1.0, 0], [0, 1]]).to("mps")
targets = torch.LongTensor([1, 0, 5]).to("mps")
F.cross_entropy(inputs, targets, ignore_index=-1, reduction="none")

import torch
import torch.nn.functional as F
inputs = torch.Tensor([[0, 1], [1.0, 0], [0, 1]]).to("cpu")
targets = torch.LongTensor([1, 0, -100]).to("cpu")
F.cross_entropy(inputs, targets, ignore_index=-1, reduction="none")

import torch
import torch.nn.functional as F
inputs = torch.Tensor([[0, 1], [1.0, 0], [0, 1]]).to("cpu")
targets = torch.LongTensor([1, 0, 5]).to("cpu")
F.cross_entropy(inputs, targets, ignore_index=-1, reduction="none")

import torch
import torch.nn.functional as F
inputs = torch.Tensor([[0, 1], [1.0, 0], [0, 1]]).to("mps")
targets = torch.LongTensor([1, 0, -100]).to("mps")
print(F.cross_entropy(inputs, targets, ignore_index=-100, reduction="none"))
print(F.cross_entropy(inputs, targets, ignore_index=-100, reduction="mean"))

import torch
import torch.nn.functional as F
inputs = torch.Tensor([[0, 1], [1.0, 0], [0, 1]]).to("cpu")
targets = torch.LongTensor([1, 0, -100]).to("cpu")
print(F.cross_entropy(inputs, targets, ignore_index=-100, reduction="none"))
print(F.cross_entropy(inputs, targets, ignore_index=-100, reduction="mean"))