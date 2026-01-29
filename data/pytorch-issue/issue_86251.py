# torch.rand(1, 64, dtype=torch.float32, device='cuda')  # Inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = -1
        self.keepdim = False

    def forward(self, input):
        dim = self.dim
        keepdim = self.keepdim
        input = torch.sub(input, torch.tensor(9, dtype=torch.float32, device=input.device))
        fn_res = torch.mean(input, dim, keepdim=keepdim)
        fn_res = torch.sub(fn_res, torch.tensor(-3, dtype=torch.float32, device=input.device))
        return fn_res

def my_model_function():
    return MyModel()

def GetInput():
    torch.random.manual_seed(56288)
    inp = torch.empty([1, 64], dtype=torch.float32, device='cuda')
    inp.uniform_(-128, 63)
    return inp

