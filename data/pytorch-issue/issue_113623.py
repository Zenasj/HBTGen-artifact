# torch.rand(B, 4, 64, 64, dtype=torch.float16)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Simplified UNet-like structure with attention (critical for SDXL UNet)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm([32, 64, 64])  # Matches spatial dimensions
        self.attn = nn.MultiheadAttention(
            embed_dim=32, 
            num_heads=4, 
            batch_first=True, 
            device="cuda", 
            dtype=torch.float16
        )
        self.conv2 = nn.Conv2d(32, 4, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        # Reshape for attention: (B, C, H, W) → (B, H*W, C)
        B, C, H, W = x.size()
        x_attn = x.view(B, C, -1).permute(0, 2, 1)
        attn_out, _ = self.attn(x_attn, x_attn, x_attn)
        # Restore spatial dimensions
        x = attn_out.permute(0, 2, 1).view(B, C, H, W)
        x = self.norm(x)
        x = self.conv2(x)
        return x

def my_model_function():
    # Initialize model with FP16 and CUDA (matches user's pipeline setup)
    model = MyModel().to("cuda").half()
    return model

def GetInput():
    # Generate input tensor matching SDXL UNet's latent space dimensions
    return torch.rand(2, 4, 64, 64, dtype=torch.float16, device="cuda")

