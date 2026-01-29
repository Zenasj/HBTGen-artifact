import torch
import torch.nn as nn

# torch.rand(B, C, H, W, dtype=torch.float32)  # Assumed input shape: (1, 512, 768) for vision-language model
# Assumptions:
# 1. Input is a 3D tensor (batch, sequence length, embedding dim)
# 2. Eager vs SPDA implementations are compared via attention modules
# 3. Output difference detected via numerical tolerance

class EagerAttention(nn.Module):
    """Stub for eager attention implementation (buggy in some cases)"""
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(768, 768)  # Placeholder layer
        
    def forward(self, x):
        # Simulate attention computation with potential numerical instability
        return self.proj(x) + 1e-6  # Added epsilon to mimic possible overflow/underflow

class SpdaAttention(nn.Module):
    """Stub for optimized SPDA attention implementation (working correctly)"""
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(768, 768)  # Match structure for comparison
        
    def forward(self, x):
        # Simulate numerically stable implementation
        return self.proj(x)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.eager_attn = EagerAttention()
        self.spda_attn = SpdaAttention()
        
    def forward(self, x):
        # Run both implementations and compare outputs
        out_eager = self.eager_attn(x)
        out_spda = self.spda_attn(x)
        
        # Check for NaNs (causing "!!!!!!!" output) and numerical differences
        has_nans = torch.isnan(out_eager).any()
        diff = torch.abs(out_eager - out_spda).max()
        
        # Return tuple indicating differences (NaNs and max difference)
        return (has_nans, diff > 1e-4)

def my_model_function():
    # Initialize model with both attention implementations
    return MyModel()

def GetInput():
    # Generate random input matching expected shape (B=1, seq_len=512, embed_dim=768)
    return torch.rand(1, 512, 768, dtype=torch.float32)

