import torch
import torch.nn as nn


class MyModule(torch.jit.ScriptModule):
#class MyModule(nn.Module):
   def __init__(self,embed_dim, num_heads):
       super(MyModule, self).__init__()
       self.mod = nn.MultiheadAttention(embed_dim, num_heads)

   def forward(self, q,k,v):
       return self.mod(q,k,v)


embed_dim = 1024
num_heads = 16
sl=30
bs=20
model = MyModule(embed_dim, num_heads).cuda()
q=torch.randn(sl,bs,embed_dim, device="cuda")
k=torch.randn(sl,bs,embed_dim, device="cuda")
v=torch.randn(sl,bs,embed_dim, device="cuda")

out = model(q,k,v)
print(out[0].size())