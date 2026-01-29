import torch
import torch.nn as nn

# torch.rand(B, C, H, W, dtype=torch.float32)  # Input shape (2, 3, 32, 32)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.attn = SelfAttentionBlock(32)
        self.conv_out = nn.Conv2d(32, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.conv_in(x)
        x = self.attn(x)
        x = self.conv_out(x)
        return x

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        query = self.query_conv(x).contiguous().reshape(B, -1, H*W)
        key   = self.key_conv(x).contiguous().reshape(B, -1, H*W)
        value = self.value_conv(x).contiguous().reshape(B, -1, H*W)

        attention = torch.bmm(query.permute(0, 2, 1), key)
        attention = torch.softmax(attention, dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.contiguous().reshape(B, C, H, W)
        return self.gamma * out + x

def my_model_function():
    return MyModel()

def GetInput():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.rand(2, 3, 32, 32, device=device, dtype=torch.float32)

