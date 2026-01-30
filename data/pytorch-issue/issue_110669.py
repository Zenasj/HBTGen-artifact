import torch.nn as nn

import random
import torch
import math

class TransformerModel(torch.nn.Module):
    def __init__(self, num_layers, model_dim, ff_dim, num_heads):
        super().__init__()
        self.layers = torch.nn.ModuleList([FastTransformerLayer(model_dim, ff_dim, num_heads) for i in range(num_layers)])

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        x = x.to_padded_tensor(padding=0)
        lens = (x!=0).sum(axis=1)[:,0]
        mean_x = x.sum(axis=1)/lens[:,None]

        return mean_x

class FastMHA(torch.nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.Q = torch.nn.Linear(model_dim, model_dim)
        self.K = torch.nn.Linear(model_dim, model_dim)
        self.V = torch.nn.Linear(model_dim, model_dim)
        self.O = torch.nn.Linear(model_dim, model_dim)
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

    def forward(self, q, k, v):
        bsz, dim = q.size(0), q.size(2)
        q,k,v = self.Q(q), self.K(k), self.V(v)
        q = q.view(bsz,-1,self.num_heads,self.head_dim).transpose(1,2)
        k = k.view(bsz,-1,self.num_heads,self.head_dim).transpose(1,2)
        v = v.view(bsz,-1,self.num_heads,self.head_dim).transpose(1,2)

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            attn_output=torch.nn.functional.scaled_dot_product_attention(q,k,v)
        attn_output = attn_output.transpose(1,2).view(bsz,-1,dim)
        attn_output = self.O(attn_output)

        return attn_output

class FastTransformerLayer(torch.nn.Module):
    def __init__(self, model_dim, ff_dim, num_heads):
        super().__init__()
        self.mha = FastMHA(model_dim, num_heads)
        self.ff = torch.nn.Sequential(torch.nn.Linear(model_dim, ff_dim), torch.nn.ReLU(), torch.nn.Linear(ff_dim, model_dim))
        self.ln1 = torch.nn.LayerNorm(model_dim)
        self.ln2 = torch.nn.LayerNorm(model_dim)

    def forward(self, x):
        x_att = self.mha(x, x, x)
        x = self.ln1(x + x_att)
        x_ff = self.ff(x)
        x = self.ln2(x + x_ff)

        return x

D=768
BS=16
MIN_SEQLEN=50
MAX_SEQLEN=300
FFDIM=2048
N_STEPS=1000

model = TransformerModel(num_layers=4, model_dim=D, ff_dim=FFDIM, num_heads=12)
model.to(device='cuda:0', dtype=torch.float16)
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.AdamW(model.parameters(),
                                lr=0.0001,
                                betas=(0.9,0.95),
                                weight_decay=0.05)
for t in range(N_STEPS):
    xlens = torch.randint(low=MIN_SEQLEN,high=MAX_SEQLEN,size=(BS,),device='cuda:0')
    x = torch.nested.nested_tensor([torch.randn((xl,D)) for xl in xlens]).to(device='cuda:0',dtype=torch.float16)
    y = torch.randn((BS,D), device=x.device, dtype=x.dtype)
    y_pred = model(x)

    loss = criterion(y_pred, y)
    print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()