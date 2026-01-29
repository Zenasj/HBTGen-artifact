# torch.rand(4, 1, 8, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.attn_with = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            add_zero_attn=True,
            batch_first=False  # Matches original code's tensor layout (seq_len, batch, ...)
        )
        self.attn_without = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            add_zero_attn=False,
            batch_first=False
        )

    def forward(self, x):
        # Dynamically generate mask based on input sequence length
        seq_len = x.size(0)
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        for i in range(seq_len):
            mask[i, :i] = False  # Allow attention to previous tokens only
        mask = mask.float() * -10000.0  # Convert to -inf mask

        # Run both attention variants and return outputs for comparison
        out_with, _ = self.attn_with(x, x, x, attn_mask=mask)
        out_without, _ = self.attn_without(x, x, x, attn_mask=mask)
        return (out_with, out_without)  # Return tuple of outputs for gradient comparison

def my_model_function():
    # Initialize with parameters from the original issue's reproduction code
    return MyModel(embedding_dim=8, num_heads=2)

def GetInput():
    # Generate random input matching the original test's dimensions (seq_len=4, batch_size=1, embedding_dim=8)
    return torch.rand(4, 1, 8, dtype=torch.float32)

