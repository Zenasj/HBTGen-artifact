# torch.rand(4, 1, 1024, 1024, dtype=torch.float32, device='cuda')
import torch
import torch.nn as nn

class MyModel(nn.Module):
    @staticmethod
    def foo(qk, batch, head, q_idx, kv_idx):
        causal_mask = q_idx <= kv_idx
        return torch.where(causal_mask, qk * 1.234, -float("inf"))
    
    def __init__(self, B=4, H=1, M=1024, N=1024, device="cuda"):
        super().__init__()
        self.B = B
        self.H = H
        self.M = M
        self.N = N
        self.device = device
    
    def forward(self, qk):
        b = torch.arange(0, self.B, device=self.device)
        h = torch.arange(0, self.H, device=self.device)
        m = torch.arange(0, self.M, device=self.device)
        n = torch.arange(0, self.N, device=self.device)
        
        # Apply vmap transformations as in original code
        score_mod = self.foo
        score_mod = torch.vmap(score_mod, in_dims=(0, None, None, None, 0))
        score_mod = torch.vmap(score_mod, in_dims=(0, None, None, 0, None))
        score_mod = torch.vmap(score_mod, in_dims=(0, None, 0, None, None))
        score_mod = torch.vmap(score_mod, in_dims=(0, 0, None, None, None))
        
        out = score_mod(qk, b, h, m, n)
        mask = torch.where(torch.isneginf(out), False, True)
        return mask

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, 1, 1024, 1024, dtype=torch.float32, device="cuda")

