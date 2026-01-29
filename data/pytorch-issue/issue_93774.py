# torch.rand(B, S, C, dtype=torch.float32)  # B=batch, S=sequence length, C=hidden size (e.g., 1024 for T5-large)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, d_model=1024, n_heads=16, dim_per_head=64):
        super().__init__()
        self.proj_layer = nn.Linear(d_model, n_heads * dim_per_head)
        self.n_heads = n_heads
        self.dim_per_head = dim_per_head

    def shape(self, x):
        # Reshape and transpose to (batch, heads, seq_length, dim_per_head)
        return x.view(x.size(0), -1, self.n_heads, self.dim_per_head).transpose(1, 2)

    def forward(self, hidden_states, key_value_states=None, past_key_value=None):
        if key_value_states is None:
            # Self-attention: project hidden_states
            proj_out = self.proj_layer(hidden_states)
            hidden_states = self.shape(proj_out)
        elif past_key_value is None:
            # Cross-attention: project key_value_states
            proj_out = self.proj_layer(key_value_states)
            hidden_states = self.shape(proj_out)
        else:
            # Handle past_key_value (e.g., in decoding)
            if key_value_states is None:
                # Self-attention with past: append new states
                new_part = self.shape(self.proj_layer(hidden_states))
                hidden_states = torch.cat([past_key_value, new_part], dim=2)
            else:
                # Cross-attention with past: reuse past_key_value
                hidden_states = past_key_value
        return hidden_states

def my_model_function():
    # T5-large parameters: d_model=1024, heads=16, dim_per_head=64 (16*64=1024)
    return MyModel()

def GetInput():
    # Batch size=8, sequence length=512 (from user's reproduction script)
    return torch.rand(8, 512, 1024, dtype=torch.float32)

