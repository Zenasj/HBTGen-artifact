# torch.rand(4, 3, 224, 224, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_float32 = nn.Conv2d(3, 1408, kernel_size=(14,14), stride=(14,14), dtype=torch.float32)
        self.conv_bfloat16 = nn.Conv2d(3, 1408, kernel_size=(14,14), stride=(14,14), dtype=torch.bfloat16)
    
    def forward(self, x):
        # Process for float32
        a_float = x[:2]
        b_float = x[2:]
        out_a_float = self.conv_float32(a_float)
        out_b_float = self.conv_float32(b_float)
        concat_float = torch.cat([out_a_float, out_b_float])
        full_float = self.conv_float32(x)
        match_float = torch.allclose(concat_float, full_float)
        
        # Process for bfloat16
        a_bfloat = a_float.to(dtype=torch.bfloat16)
        b_bfloat = b_float.to(dtype=torch.bfloat16)
        x_bfloat = x.to(dtype=torch.bfloat16)
        out_a_bfloat = self.conv_bfloat16(a_bfloat)
        out_b_bfloat = self.conv_bfloat16(b_bfloat)
        concat_bfloat = torch.cat([out_a_bfloat, out_b_bfloat])
        full_bfloat = self.conv_bfloat16(x_bfloat)
        match_bfloat = torch.allclose(concat_bfloat, full_bfloat)
        
        return torch.tensor([match_float, match_bfloat], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, 3, 224, 224, dtype=torch.float32)

