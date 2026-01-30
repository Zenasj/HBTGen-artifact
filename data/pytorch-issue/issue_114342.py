import torch.nn as nn

import torch
def fn(input_0, input_1):
  input_size = 10
  hidden_size = 20
  num_layers = 2
  arg_class = torch.nn.GRU(input_size, hidden_size, num_layers=num_layers, device='cpu', dtype=torch.float64)
  return arg_class(input_0, input_1)

input_tensor_0 = torch.empty([5, 3, 10], dtype=torch.float64, requires_grad=True)
input_tensor_1 = torch.empty([2, 3, 20], dtype=torch.float64, requires_grad=True)
torch.autograd.gradcheck(fn, (input_tensor_0, input_tensor_1))

import torch
def fn(input_0, input_1):
  input_size = 10
  hidden_size = 20
  num_layers = 2
  arg_class = torch.nn.RNN(input_size, hidden_size, num_layers=num_layers, device='cpu', dtype=torch.float64)
  return arg_class(input_0, input_1)

input_tensor_0 = torch.empty([5, 3, 10], dtype=torch.float64, requires_grad=True)
input_tensor_1 = torch.empty([2, 3, 20], dtype=torch.float64, requires_grad=True)
torch.autograd.gradcheck(fn, (input_tensor_0, input_tensor_1))