import torch

def fn(x, y):
    t = torch.bitwise_and(x, y) # ✔️ works fine w/o this line
    return torch.clamp_max(t, y)
    # ❌ return torch.max(t, y)

x = torch.rand([5, 10, 1]).to(torch.int8)
y = torch.rand([10, 1]).to(torch.int8)

# ✔️ x, y are cuda tensors

# ❌ torch.bool, torch.int8, torch.int16, torch.uint8, torch.uint16
# ✔️ torch.int32, torch.int64

ret_eager = fn(x, y)
print('==== Eager mode OK! ====')

compiled = torch.compile(fn)
print('==== torchcomp compilation OK! ====')

ret_compiled = compiled(x, y)
print('==== torchcomp mode OK! ====')