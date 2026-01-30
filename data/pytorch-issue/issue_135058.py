import torch.nn as nn

import torch

DIM = 4096
INPUT_SIZE1 = 32
INPUT_SIZE2 = 16

class LinearNet(torch.nn.Module):
   def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(DIM, DIM, bias=False)

   def forward(self, x):
        x = self.fc1(x)
        return x

input1 = torch.randn(size=(INPUT_SIZE1, DIM))
input2 = torch.randn(size=(INPUT_SIZE2, DIM))

with torch.no_grad():
    model = LinearNet()
    model =  torch.ao.quantization.quantize_dynamic(model,{torch.nn.Linear})
    
    model(input1)   # this goes to ACL lowp_gemm
    print("="*50)
    model(input2)   # this goes to gemm:jit without this PR, and to ACL with this PR