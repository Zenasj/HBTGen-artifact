import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):     
        return torch.argmax(input=torch.fliplr(input=input)) 

input1 = torch.rand([8, 6, 2, 6, 6, 1, 1, 4, 1], dtype=torch.float32)
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