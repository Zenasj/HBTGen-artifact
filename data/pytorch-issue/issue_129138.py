import torch
import torch.nn as nn

class M(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 64, bias=False)
                self.bias1 = torch.randn(64).bfloat16()  # if the bias is not bf16, we will crash

            def forward(self, x):
                return self.linear1(x) + self.bias1