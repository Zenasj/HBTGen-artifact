# torch.rand(B, 1024, dtype=torch.int64)
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_embd, n_heads, **kwargs):
        super().__init__()
        self.d_head = d_embd // n_heads
        self.attn_proj = nn.Linear(d_embd, 3 * d_embd)
        self.out_proj = nn.Linear(d_embd, d_embd)
    
    def forward(self, x_BTE):
        qkv = self.attn_proj(x_BTE).split(x_BTE.size(-1), -1)
        split_attn_head = lambda z: z.unflatten(-1, [-1, self.d_head]).transpose(1, 2)
        q_BHTD, k_BHTD, v_BHTD = map(split_attn_head, qkv)
        o_BHTD = F.scaled_dot_product_attention(q_BHTD, k_BHTD, v_BHTD, dropout_p=0.0, is_causal=True)
        o_BTE = o_BHTD.transpose(1, 2).flatten(-2)
        y_BTE = self.out_proj(o_BTE)
        return y_BTE

class GPTBlock(nn.Module):
    def __init__(self, d_embd, **kwargs):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_embd)
        self.attn = CausalSelfAttention(d_embd, **kwargs)
        self.ffn_norm = nn.LayerNorm(d_embd)
        self.ffn = nn.Sequential(
            nn.Linear(d_embd, 4 * d_embd),
            nn.GELU(),
            nn.Linear(4 * d_embd, d_embd)
        )
    
    def forward(self, x_BTE):
        x_BTE = x_BTE + self.attn(self.attn_norm(x_BTE))
        y_BTE = x_BTE + self.ffn(self.ffn_norm(x_BTE))
        return y_BTE

class MyModel(nn.Module):
    def __init__(self, vocab_size, max_seq_len, n_layers, d_embd, **kwargs):
        super().__init__()
        self.tok_embd = nn.Embedding(vocab_size, d_embd)
        self.pos_embd = nn.Embedding(max_seq_len, d_embd)
        self.tsfmr_blks = nn.ModuleList([GPTBlock(d_embd, **kwargs) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_embd)
    
    def forward(self, idx_BT):
        pos_T = torch.arange(idx_BT.size(1), dtype=torch.int64, device=idx_BT.device)
        x_BTE = self.tok_embd(idx_BT) + self.pos_embd(pos_T).unsqueeze(0)
        
        for tsfmr_blk in self.tsfmr_blks:
            x_BTE = tsfmr_blk(x_BTE)
        
        x_BTE = self.out_norm(x_BTE)
        logits_BTV = x_BTE @ self.tok_embd.weight.T
        return logits_BTV

def my_model_function():
    return MyModel(
        vocab_size=50304,
        max_seq_len=1024,
        n_layers=48,
        d_embd=1600,
        n_heads=25  # Required for CausalSelfAttention
    )

def GetInput():
    return torch.randint(50304, (1, 1024), dtype=torch.int64)

