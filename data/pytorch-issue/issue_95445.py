# torch.rand(B, 1024, dtype=torch.int64)  # Input shape: (batch_size, sequence_length)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, vocab_size=50257, block_size=1024, n_embd=384, n_head=6, n_layer=6, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)  # Tied to tok_emb weights

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = CausalSelfAttention(n_embd, head_size, n_head, dropout)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, head_size, n_head, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size * n_head, bias=False)
        self.query = nn.Linear(n_embd, head_size * n_head, bias=False)
        self.value = nn.Linear(n_embd, head_size * n_head, bias=False)
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        head_size = self.head_size
        n_head = self.n_head
        # Merge head乘数 into batch dimension
        k = self.key(x).view(B, T, n_head, head_size).transpose(1, 2)
        q = self.query(x).view(B, T, n_head, head_size).transpose(1, 2)
        v = self.value(x).view(B, T, n_head, head_size).transpose(1, 2)
        # Causal attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / (head_size ** 0.5))
        att = att.masked_fill(torch.tril(torch.ones(T, T, device=x.device)) == 0, float('-inf'))
        att = att.softmax(dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, n_head * head_size)
        return self.proj(y)

def my_model_function():
    return MyModel()

def GetInput():
    # Use batch size 1 as in original example
    return torch.randint(0, 50257, (1, 1024), dtype=torch.int64, device="cuda")

