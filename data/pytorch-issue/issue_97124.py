import torch
def forward():
    a = torch.zeros([1, 2], dtype=torch.int32)
    a = a + a
    b = a.to(dtype=torch.float32)
    return b * 0.8

fn_compiled = torch.compile(forward)
print(fn_compiled())