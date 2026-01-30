py
import torch

if __name__ == '__main__':
    a = torch.randn(32, 100, 3, requires_grad=True)

    b = torch.sum(a, dim=(), keepdim=False)
    print(b)
    b.backward()
    c = torch.mean(a, dim=(), keepdim=False)
    print(c)
    c.backward()