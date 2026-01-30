import torch
import torch.nn as nn

def test_copy_int(self): 
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.b = torch.ones(3)
        
            def forward(self, x):
                seq_len = 2
                self.b.copy_(seq_len)
                return x + self.b
        
        M()(torch.ones(3))
        torch.compile(M())(torch.ones(3))