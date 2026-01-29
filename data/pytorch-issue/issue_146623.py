# torch.rand(16, 16, 80, dtype=torch.float32)  # Input shape for q (first element of input tuple)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs):
        q, k, v, attention_mask = inputs
        # Compute CPU output
        oo_cpu = F.scaled_dot_product_attention(
            q.to("cpu"),
            k.to("cpu"),
            v.to("cpu"),
            attention_mask.to("cpu"),
            dropout_p=0.0
        )
        
        # Compute MPS output if available
        if torch.backends.mps.is_available():
            oo_mps = F.scaled_dot_product_attention(
                q.to("mps"),
                k.to("mps"),
                v.to("mps"),
                attention_mask.to("mps"),
                dropout_p=0.0
            )
            # Compare outputs with tolerance
            return torch.allclose(oo_cpu, oo_mps.to("cpu"), atol=1e-5)
        else:
            # Return True if MPS not available (assumed compatible)
            return torch.tensor(True, dtype=torch.bool)
        
def my_model_function():
    return MyModel()

def GetInput():
    head_num, seq_len, embed_dim = 16, 16, 80
    q = torch.randn(head_num, seq_len, embed_dim)
    k = torch.randn(head_num, seq_len, embed_dim)
    v = torch.randn(head_num, seq_len, embed_dim)
    attention_mask = torch.ones(1, seq_len, seq_len)
    return (q, k, v, attention_mask)

