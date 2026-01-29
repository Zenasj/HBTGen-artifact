# torch.rand(10, 5, 24), torch.rand(10,5,24), torch.rand(10,5,12)  # Input is a tuple of (query, key, value)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=24,  # Query/Key dimension
            num_heads=1,
            vdim=12        # Value dimension (different from embed_dim)
        )
        
    def forward(self, inputs):
        query, key, value = inputs
        attn_output, _ = self.mha(query, key, value)
        return attn_output

def my_model_function():
    return MyModel()

def GetInput():
    seq_len = 10
    batch_size = 5
    embed_dim = 24
    value_dim = 12
    # Create tensors matching the expected input shapes
    query = torch.randn(seq_len, batch_size, embed_dim)
    key = torch.randn(seq_len, batch_size, embed_dim)  # Same as embed_dim
    value = torch.randn(seq_len, batch_size, value_dim)
    return (query, key, value)

