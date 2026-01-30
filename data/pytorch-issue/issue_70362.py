import torch
import torch.nn as nn

torch.set_default_dtype(torch.double)

rnn = nn.LSTMCell(10, 20)
input = torch.randn(3, 10, requires_grad=True)
hx = (torch.randn(3, 20, requires_grad=True), torch.randn(3, 20, requires_grad=True))

# Works
p = tuple(rnn.parameters())
inp = (input,) + hx + p

def fn(*inp):
    (input, hx_0, hx_1) = inp[:3]
    return rnn(input, (hx_0, hx_1))

torch.autograd.gradcheck(fn, inp)

# Doesn't work (the way gradcheck in test_modules passes it's inputs)
p = tuple(rnn.parameters())
inp = (input, hx) + p

def fn(*inp):
    (input, hx) = inp[:2]
    return rnn(input, hx)

torch.autograd.gradcheck(fn, inp)