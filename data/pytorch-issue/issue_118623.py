import torch
import torch.nn as nn
import os, pickle

with open('inputs.pkl', 'rb') as f:
    inputs = pickle.load(f) # 'p1': ..., 'p2': ...

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, out):

        out = torch.special.digamma(out=out, input=inputs[0]['input'])        
        out = torch.argmax(input=out, )        
        return out

input1 = torch.rand([8, 6, 8, 6, 6, 1, 3], dtype=torch.float32)
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