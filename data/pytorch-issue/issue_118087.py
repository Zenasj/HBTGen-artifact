import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):    
        res = torch.nn.Linear(in_features=2,out_features=2)(input=input)        
        return res

x = torch.rand([8, 6, 6, 1, 3, 4, 1, 5, 2], dtype=torch.float32)
model = Model().to(torch.device('cpu'))
eag = model(x)
opt = torch.compile(model.forward, mode='default')(x)

same_val = torch.allclose(eag.to('cpu'), 
                            opt.to('cpu'), 
                            rtol=1e-3, atol=1e-3, 
                            equal_nan=True)
if same_val == False : 
        raise ValueError('diff value')