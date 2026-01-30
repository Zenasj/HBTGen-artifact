import torch
import torch.nn as nn
import os, pickle


with open(os.path.join('/home/exp_data/torch-2','inputs.pkl'), 'rb') as f:
    inputs = pickle.load(f) # 'p1': ..., 'p2': ...


class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):  
        return torch.nn.functional.kl_div(input=input, reduction='none', target=inputs[0]['target'])        

x = torch.rand([1, 7, 3, 9, 9, 7], dtype=torch.float32)


model = Model().to(torch.device('cpu'))
eag = model(x)
opt = torch.compile(model.forward, mode='default')(x)

same_val = torch.allclose(eag.to('cpu'), 
                            opt.to('cpu'), 
                            rtol=1e-3, atol=1e-3, 
                            equal_nan=True)
if same_val == False : 
        raise ValueError('diff value')