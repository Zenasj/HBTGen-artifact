import torch
import torch.nn as nn

class fused(torch.nn.Module):
    def forward(self, a, b):
        fused_1 = self.fused_1(a, b);
        relu = fused_1.relu()
        fused_0 = self.fused_0(fused_1, relu)
        return (fused_0, fused_1)

    class fused_0(torch.nn.Module):
        def forward(self, add_2, relu):
            ... # Logic after relu
            return add_4 
    
    class fused_1(torch.nn.Module):
        def forward(self, a, b):
            ... # Logic before relu, `add_1` is only exposed within this submodule.
            return add_2