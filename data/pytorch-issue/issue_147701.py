import torch.nn as nn

python
import torch
from torch.nn.attention.flex_attention import flex_attention

B  =    1
L  = 1024
H  =    1
Ev =   16
E  =    8 # breaks for any E != Ev

@torch.compile
def fct(Q,K,V):
    out = flex_attention(Q,K,V) # (B,H,L,Ev)
    out = out.transpose(1,2)    # (B,L,H,Ev)
    out = out.reshape(B,L,H*Ev) # (B,L,H*Ev)
    return out

Q = torch.randn(B,H,L,E ).cuda()
K = torch.randn(B,H,L,E ).cuda()
V = torch.randn(B,H,L,Ev).cuda()

out = fct(Q,K,V)
print(out.shape)