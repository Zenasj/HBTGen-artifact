# torch.randint(0, 8096, (B, 64), dtype=torch.long)  # Input shape (B, T=64), vocab_size=8096
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTConfig:
    def __init__(self, block_size, vocab_size, n_layer, n_head, n_embd, dropout, bias):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias

class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_emb(idx)
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)
        pos_emb = self.pos_emb(pos)  # (1, T, C)
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return loss
        return logits

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.dropout,
            batch_first=True,
            bias=config.bias
        )

    def forward(self, x):
        B, T, C = x.size()
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_output, _ = self.attn(
            query=x,
            key=x,
            value=x,
            attn_mask=mask,
            need_weights=False
        )
        return attn_output

def my_model_function():
    config = GPTConfig(
        block_size=256,
        vocab_size=8096,
        n_layer=2,
        n_head=2,
        n_embd=128,
        dropout=0.0,
        bias=False
    )
    return MyModel(config)

def GetInput():
    return torch.randint(0, 8096, (2, 64), dtype=torch.long)

