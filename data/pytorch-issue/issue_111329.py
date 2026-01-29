# torch.rand(T, B, H, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def true_fn(self, carry, inp):
        return torch.sigmoid(inp) + carry
    
    def false_fn(self, carry, inp):
        return torch.zeros_like(carry)
    
    def forward(self, input):
        dim0 = input.size(0)
        pad_amount = 10 - dim0
        input_padded = torch.nn.functional.pad(input, (0, 0, 0, 0, 0, pad_amount))
        outs = []
        in0 = input[0, :]
        for t in range(10):
            condition = t < dim0
            in0 = torch.cond(condition, self.true_fn, self.false_fn, (in0, input_padded[t, :]))
            outs.append(in0)
        return torch.stack(outs)

def my_model_function():
    return MyModel()

def GetInput():
    T = 8  # as per the example input (t_rnd=8)
    B = 64
    H = 140
    return torch.rand(T, B, H)

