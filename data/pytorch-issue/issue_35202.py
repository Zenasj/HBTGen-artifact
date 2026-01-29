# torch.rand(1, 1, 480, 640, dtype=torch.float32).cuda()
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Problematic grid with extreme values causing illegal memory access
        coords = torch.tensor([[-10059144, 67680944], 
                              [67680944, 67680944]], dtype=torch.float32)
        coords = coords.unsqueeze(0).unsqueeze(0).cuda()  # Shape (1,1,2,2)
        self.register_buffer('grid', coords)

    def forward(self, x):
        return F.grid_sample(x, self.grid, align_corners=False)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 480, 640, dtype=torch.float32).cuda()

