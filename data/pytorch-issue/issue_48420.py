# Input: (torch.rand(1,2,5), torch.rand(3,5,4))

import torch
from torch import nn

def extended_einsum(eq, *tensors):
    lhs, rhs = eq.split("->")
    mod_ops, mod_t = [], []
    for op, t in zip(lhs.split(","), tensors):
        mod_op = ""
        for i, c in reversed(list(enumerate(op))):
            if t.shape[i] == 1:
                t = t.squeeze(dim=i)
            else:
                mod_op = c + mod_op
        mod_ops.append(mod_op)
        mod_t.append(t)
    m_lhs = ",".join(mod_ops)
    r_keep_dims = set("".join(mod_ops))
    m_rhs = "".join(c for c in rhs if c in r_keep_dims)
    m_eq = f"{m_lhs}->{m_rhs}"
    mod_r = torch.einsum(m_eq, *mod_t)
    for i, c in enumerate(rhs):
        if c not in r_keep_dims:
            mod_r = mod_r.unsqueeze(dim=i)
    return mod_r

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, inputs):
        x, y = inputs
        matmul_result = x @ y
        einsum_result = extended_einsum('bij,bjk->bik', x, y)
        return torch.tensor(torch.allclose(matmul_result, einsum_result), dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.rand(1, 2, 5)
    y = torch.rand(3, 5, 4)
    return (x, y)

