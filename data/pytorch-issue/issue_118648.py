import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, out, input1, other, input2):
        res = torch.minimum(out=out, input=input1,other=other)        
        res = torch.amin(out=res, dim=0,input=input2,keepdim=False)        
        return res

input1 = torch.rand([1, 1, 1], dtype=torch.float32)
input2 = input1.clone()
input3 = torch.rand([1, 1, 1], dtype=torch.float32)
input4 = torch.rand([1, 1, 1], dtype=torch.float32)
input5 = torch.rand([1, 1, 1], dtype=torch.float32)

model = Model().to(torch.device('cpu'))
eag = model(input1, input3, input4, input5)
opt = torch.compile(model.forward)(input2, input3, input4, input5)
print(f"shape in eagermode: {eag.shape}")
print(f"shape in torch.compile: {opt.shape}")