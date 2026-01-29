# torch.randint(0, 50257, (B, T), dtype=torch.long)  # Inferred input shape (B=batch, T=sequence length)
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelA(nn.Module):
    def __init__(self, n_embd=768, bias=False):
        super().__init__()
        self.n_head = 6
        self.embd = nn.Embedding(50257, n_embd)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.dropout = 0.0  # No dropout

    def forward(self, x):
        x = self.embd(x)
        B, T, C = x.size()
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            query=q, key=k, value=v,  # Named parameters (causes error when compiled)
            attn_mask=None, dropout_p=self.dropout, is_causal=True
        )
        return y

class ModelB(nn.Module):
    def __init__(self, n_embd=768, bias=False):
        super().__init__()
        self.n_head = 6
        self.embd = nn.Embedding(50257, n_embd)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.dropout = 0.0

    def forward(self, x):
        x = self.embd(x)
        B, T, C = x.size()
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(  # Positional arguments (works when compiled)
            q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True
        )
        return y

class MyModel(nn.Module):
    def __init__(self, n_embd=768, bias=False):
        super().__init__()
        self.model_a = ModelA(n_embd, bias)  # Model using named parameters
        self.model_b = ModelB(n_embd, bias)  # Model using positional arguments

    def forward(self, x):
        out_a = self.model_a(x)
        out_b = self.model_b(x)
        # Compare outputs using element-wise tolerance
        diff = torch.isclose(out_a, out_b, atol=1e-6)
        return torch.all(diff)  # Returns tensor(bool) indicating equality

def my_model_function():
    return MyModel(n_embd=768, bias=False)

def GetInput():
    batch_size = 2
    seq_len = 1024
    return torch.randint(0, 50257, (batch_size, seq_len), dtype=torch.long)

