import torch
import torch.nn as nn
import os, pickle

with open(os.path.join('inputs.pkl'), 'rb') as f:
    inputs = pickle.load(f) # 'p1': ..., 'p2': ...

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, target, input):
        target = torch.nn.functional.huber_loss(target=target, delta=1.0,input=input)        
        target = torch.nn.functional.l1_loss(target=target, input=inputs[1]['input'],reduction='none')        
        return target
inf = float('inf')
nan = float('nan')
is_valid = True
input1 = inputs['target']
input2 = input1.clone()
input3 = torch.rand([1, 7, 6, 6, 2, 5], dtype=torch.float32)

model = Model().to(torch.device('cpu'))
eag = model(input1, input3)
opt = torch.compile(model.forward)(input2, input3)

same_val = torch.allclose(eag.to('cpu'), 
                            opt.to('cpu'), 
                            rtol=1e-3, atol=1e-3, 
                            equal_nan=True)
if same_val == False : 
        raise ValueError('diff value')