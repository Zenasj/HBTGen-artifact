import torch
from torch._subclasses.fake_tensor import FakeTensorMode

if __name__ == '__main__':
    torch.cuda.set_device(0)
    # torch.set_default_device('cuda')
    with FakeTensorMode():
        p = torch.randn(4, 2, requires_grad=True, device='cuda')
        x = torch.randn(8, 4, device='cuda')
        y = torch.mm(x, p).square().sum()
        y.backward()