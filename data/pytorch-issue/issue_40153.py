import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.rand(1, 2, 10, 10, dtype=torch.float32)
class MyModel(nn.Module):
    def forward(self, input):
        pos = input.permute(0, 2, 3, 1)
        H, W = input.size(2), input.size(3)
        grid1 = (pos + 0.5) / H * 2 - 1  # align_corners=False grid (pixel center)
        grid2 = pos / (H - 1) * 2 - 1    # align_corners=True grid (pixel index)

        # Case1: grid1 with align_corners=False (should be small)
        out1 = F.grid_sample(input, grid1, align_corners=False)
        # Case2: grid2 with align_corners=True (should be small)
        out2 = F.grid_sample(input, grid2, align_corners=True)
        # Case3: grid1 with align_corners=True (should be large)
        out3 = F.grid_sample(input, grid1, align_corners=True)
        # Case4: grid2 with align_corners=False (should be large)
        out4 = F.grid_sample(input, grid2, align_corners=False)

        diff1 = torch.abs(out1 - input).sum()
        diff2 = torch.abs(out2 - input).sum()
        diff3 = torch.abs(out3 - input).sum()
        diff4 = torch.abs(out4 - input).sum()
        
        return diff1, diff2, diff3, diff4

def my_model_function():
    return MyModel()

def GetInput():
    B, C, H, W = 1, 2, 10, 10
    input = torch.zeros(B, C, H, W, dtype=torch.float32)
    for i in range(H):
        for j in range(W):
            input[0, 0, i, j] = j  # x-coordinate (columns)
            input[0, 1, i, j] = i  # y-coordinate (rows)
    return input

