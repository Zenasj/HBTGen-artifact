import torch
import torch.nn as nn

class MiniAttention(torch.nn.Module):
    def __init__(self, E):
        super(MiniAttention, self).__init__()
        self.k_proj = nn.Linear(E, E)
        self.q_proj = nn.Linear(E, E)
        self.v_proj = nn.Linear(E, E)
    
    def forward(self, x):
        B, E = x.size(0), x.size(2)
        k, q, v, = self.k_proj(x), self.q_proj(x), self.v_proj(x)
        # kT = torch.nested.as_nested_tensor([y.transpose(-1, -2) for y in k.unbind(0)]) # works
        kT = k.transpose(-1, -2) # does not work
        attn_weights =  q @ kT
        attn_output = attn_weights @ v
        attn_output = torch.nested.to_padded_tensor(attn_output, padding=0.).mean(dim=1) # mean pooling
        return attn_output

L, E = (1, 2), 3
x = torch.nested.nested_tensor([
    torch.rand(L[0], E), 
    torch.rand(L[1], E)
])

model = MiniAttention(E)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
output = model(x)
loss = (0. - output).pow(2).sum()
loss.backward() # RuntimeError: Expected nested_tensor_impl_is_contiguous(nt_grad_output) to be true, but got false.

print(model.k_proj.weight.grad) # None
print(model.k_proj.bias.grad) # None