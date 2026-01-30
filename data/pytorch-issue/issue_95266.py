model = Model().cuda()
batch_size = 2
x = torch.randint(0, 50257, (batch_size, 1024)).cuda()
y = model(x)

model = torch.compile(model)
_ = model(x)

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, n_embd=768, bias=False):
        super().__init__()
        self.n_head = 6
        self.embd = nn.Embedding(50257, 768)
        self.c_attn = nn.Linear(in_features=n_embd, out_features=3 * n_embd, bias=bias)
        self.dropout = 0

    def forward(self, x):
        x = self.embd(x)
        (B, T, C) = x.size()
        q, k, v = self.c_attn(x).chunk(chunks=3, dim=-1)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True
        )
        return y

model = Model().cuda()
batch_size = 2
x = torch.randint(0, 50257, (batch_size, 1024)).cuda()
model = torch.compile(model)
_ = model(x)
y = model(x)