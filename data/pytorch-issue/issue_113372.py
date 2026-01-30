import torch

def test_constrain_size_in_eager(self):
        def fn(x, y):
            n = x.max().item()
            torch._constrain_as_size(n, max=5)
            return y + n

        ep = export(fn, (torch.randint(1, 2, (2, 2)), torch.randint(3, 5, (2, 3))))
        ep(torch.randint(6, 7, (2, 2)), torch.randint(3, 5, (2, 3)))