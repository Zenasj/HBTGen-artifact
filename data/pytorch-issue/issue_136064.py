import torch

def fn(x, y):
    view = x.view(torch.float32)
    view.copy_(y)

def get_inp():
    x = torch.zeros(4, 4, device="cuda")
    y = torch.ones(4, 4, device="cuda")
    return x, y

xc, yc = get_inp()
xe, ye = get_inp()
fn(xe, ye)
torch.compile(fn)(xc, yc)
assert torch.all(xe == xc)