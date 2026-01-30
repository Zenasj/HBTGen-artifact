import torch

def fn(inputs, dim):
    res = torch.sum(inputs, dim)
    return res

if __name__ == "__main__":
    inputs = torch.randn(128, 5, 24, 24)
    dim = tuple([-1, 1, 0, 2])
    compl_fn = torch.compile(fn, dynamic=True)
    res = compl_fn(inputs, dim)
    print(res)