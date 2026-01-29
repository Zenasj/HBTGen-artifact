# torch.rand(1, S, dtype=torch.long)  # S is sequence length (e.g., 1, 2, 3...)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, max_output_patches=100, output_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(1000, output_dim).to(torch.bfloat16).cuda()
        self.cross_attn = nn.MultiheadAttention(
            output_dim, 8, batch_first=True, device="cuda", dtype=torch.bfloat16
        )
        # Predefined encoder outputs (part of model state from setup_cache)
        self.encoder_outputs = torch.randn(
            1, max_output_patches, output_dim, 
            device="cuda", dtype=torch.bfloat16
        )
        self.encoder_cache_pos = torch.arange(max_output_patches).cuda()

    def forward(self, input_ids):
        # Embedding layer
        x = self.embedding(input_ids)
        # Cross-attention using encoder outputs as key/value
        attn_output, _ = self.cross_attn(
            x, 
            self.encoder_outputs, 
            self.encoder_outputs
        )
        return attn_output

def my_model_function():
    # Matches configuration from issue's encoder_args
    return MyModel(max_output_patches=256, output_dim=768)

def GetInput():
    # Example input size where slowdown occurs (n=2)
    return torch.full(
        (1, 2), 1, 
        dtype=torch.long, 
        device="cuda"
    )

