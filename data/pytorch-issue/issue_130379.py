import torch
import torch.nn as nn

class M(torch.nn.Module):
    def forward(self, x, y):
        with torch.no_grad():
            c = torch.tensor(4)
            z = c + x + y 
        
        return z * z

torch.export.export(M(), (torch.ones(4), torch.ones(4)))