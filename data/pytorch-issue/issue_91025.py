# torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256), torch.rand(1, 1, 256, 256), torch.rand(1, 2, 256, 256), dtype=torch.float32
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_dec_fg1 = nn.Conv2d(9, 3, kernel_size=3, padding=1)  # 9 channels from concatenated inputs
        # Placeholder for upstream layers (assumed to handle input concatenation)
    
    def forward(self, inputs):
        image, bg, seg, multi_fr = inputs
        x = torch.cat([image, bg, seg, multi_fr], dim=1)  # Concatenate along channel axis
        out_dec_fg = self.model_dec_fg1(x)
        # Dummy second output to match original return signature (alpha_pred, fg_pred)
        return out_dec_fg, torch.zeros_like(out_dec_fg)

def my_model_function():
    return MyModel()

def GetInput():
    return (
        torch.rand(1, 3, 256, 256),   # image
        torch.rand(1, 3, 256, 256),   # bg
        torch.rand(1, 1, 256, 256),   # seg
        torch.rand(1, 2, 256, 256)    # multi_fr
    )

