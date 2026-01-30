import random

py
import torch.nn as nn
import torch.nn.functional as F
import torch


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(128, embedding_dim=128)
        self.cross_attn = MultiheadAttention(128, 8)

    def forward(self, tokens):
        embed = self.tok_emb(tokens)
        z = torch.randn(embed.shape[0], 4, embed.shape[2], device=embed.device, dtype=embed.dtype)
        logits = self.cross_attn(embed, z, z)
        return logits.mean()


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, heads: int):
        super().__init__()

        self.embed_size = embed_dim
        self.num_heads = heads
        self.head_dim = embed_dim // heads

        assert self.head_dim * heads == embed_dim, "Embedding size needs to be divisible by heads"

        self.q_proj = nn.Linear(self.embed_size, self.embed_size)
        self.k_proj = nn.Linear(self.embed_size, self.embed_size)
        self.v_proj = nn.Linear(self.embed_size, self.embed_size)

    def forward(self, query, key, value):
        N = query.shape[0]

        q = query.view(N, -1, self.num_heads, self.head_dim).swapaxes(1, 2)
        k = key.view(N, -1, self.num_heads, self.head_dim).swapaxes(1, 2)
        v = value.view(N, -1, self.num_heads, self.head_dim).swapaxes(1, 2)

        # Without compile: torch.bfloat16 torch.bfloat16 torch.bfloat16
        # With compile:    torch.bfloat16 torch.float32 torch.float32
        print(q.dtype, k.dtype, v.dtype)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.swapaxes(1, 2).reshape(N, -1, self.embed_size)
        return attn


model = VAE()
model = VAE().to(device="cuda", dtype=torch.bfloat16)

# Without compile, this runs fine
model = torch.compile(model)

input = torch.randint(0, 128, (512, 128), device=torch.device("cuda"))

torch.set_default_dtype(torch.bfloat16)  # < -------- THIS LINE IS PROBLEMATIC?
loss = model(input)
loss.backward()

py
torch.set_default_dtype(torch.bfloat16)

def forward(self, primals_1, primals_2):
    embedding = torch.ops.aten.embedding.default(primals_1, primals_2);  primals_1 = None
    inductor_seeds_default = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))
    inductor_lookup_seed_default = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
    inductor_random_default = torch.ops.prims.inductor_random.default([512, 4, 128], inductor_lookup_seed_default, 'randn');  inductor_lookup_seed_default = None
    view = torch.ops.aten.view.default(embedding, [512, -1, 8, 16]);  embedding = None
    permute = torch.ops.aten.permute.default(view, [0, 2, 1, 3]);  view = None
    view_1 = torch.ops.aten.view.default(inductor_random_default, [512, -1, 8, 16])
    permute_1 = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
    view_2 = torch.ops.aten.view.default(inductor_random_default, [512, -1, 8, 16]);  inductor_random_default = None
    permute_2 = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
    return (permute, permute_1, permute_2, primals_2)