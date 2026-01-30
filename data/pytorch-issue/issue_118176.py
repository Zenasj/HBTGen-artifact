import numpy as np

import torch
import torch.nn as nn
import os, pickle

with open('inputs.pkl', 'rb') as f:
    inputs = pickle.load(f) # 'p1': ..., 'p2': ...

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, out,): 
        out = torch.special.sinc(out=out, input=inputs[0]['input'])        
        return out

input1 = torch.rand([4, 1, 3, 2, 8, 2, 6], dtype=torch.float32)
input2 = input1.clone()
model = Model().to(torch.device('cpu'))
eag = model(input1)
opt = torch.compile(model.forward)(input2)
same_val = torch.allclose(eag.to('cpu'), 
                            opt.to('cpu'), 
                            rtol=1e-3, atol=1e-3, 
                            equal_nan=True)
if same_val == False : 
        raise ValueError('diff value')

inp = torch.linspace(0, 10000, 2**14, dtype=torch.float16)
(Pdb) (torch.sinc(inp.to(dtype=torch.float64)) - torch.compile(torch.sinc)(inp)).abs().mean()
tensor(5.1601e-08, dtype=torch.float64)
(Pdb) (torch.sinc(inp.to(dtype=torch.float64)) - torch.sinc(inp)).abs().mean()
tensor(6.6042e-05, dtype=torch.float64)