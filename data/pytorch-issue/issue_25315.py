import torch.nn as nn

model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Sigmoid()).to(gpu)
model_in = torch.rand(3, 4, requires_grad=True, device=gpu)
model_out = model(model_in)
model_first_grad = torch.autograd.grad(model_out.sum(), model_in, create_graph=True)[0]
model_second_grad = torch.autograd.grad(model_first_grad.sum(), model_in)[0]
print(model_second_grad) # this returns the second grad

import torch
gpu = torch.device('cuda')

rnn_model = torch.nn.RNN(4, 4).to(gpu)
rnn_in = torch.rand(1, 3, 4, requires_grad=True, device=gpu)
rnn_out, _ = rnn_model(rnn_in)
rnn_first_grad = torch.autograd.grad(rnn_out.sum(), rnn_in, create_graph=True)[0]
rnn_second_grad = torch.autograd.grad(rnn_first_grad.sum(), rnn_in)[0]
print(rnn_second_grad) # this never runs

import torch
gpu = torch.device('cuda')

with torch.backends.cudnn.flags(enabled=False):
    rnn_model = torch.nn.LSTM(4, 4).to(gpu)
    rnn_in = torch.rand(1, 3, 4, requires_grad=True, device=gpu)
    rnn_out, _ = rnn_model(rnn_in)
    rnn_first_grad = torch.autograd.grad(rnn_out.sum(), rnn_in, create_graph=True)[0]
    rnn_second_grad = torch.autograd.grad(rnn_first_grad.sum(), rnn_in)[0]
    print(rnn_second_grad) # this never runs

import torch
gpu = torch.device('cuda')

with torch.backends.cudnn.flags(enabled=False):
    rnn_model = torch.nn.GRU(4, 4).to(gpu)
    rnn_in = torch.rand(1, 3, 4, requires_grad=True, device=gpu)
    rnn_out, _ = rnn_model(rnn_in)
    rnn_out = rnn_out.pow(2)  # to ensure hessian exists
    rnn_first_grad = torch.autograd.grad(rnn_out.sum(), rnn_in, create_graph=True)[0]
    rnn_second_grad = torch.autograd.grad(rnn_first_grad.sum(), rnn_in)[0]
    print(rnn_second_grad) # this never runs