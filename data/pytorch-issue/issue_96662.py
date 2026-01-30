import torch
x = torch.rand([2,1,1,1], device='cuda')
def forward():
    a = x.argmax(3) # [2,1,1]
    b = a.max(2).values #[2,1]
    c = b.sum(0) # [1]
    return torch.add(b, c)
fn_compiled = torch.compile(forward)
print(fn_compiled())