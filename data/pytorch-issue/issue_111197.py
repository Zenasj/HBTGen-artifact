import torch


def fn(inputs, dim):
    res = torch.amax(inputs, dim)
    return res


if __name__ == "__main__":
    inputs = torch.randn(2, 3, 4, 5)
    dim = tuple([0, 3, 2, 1])
    compl_fn = torch.compile(fn, dynamic=True)
    res = compl_fn(inputs, dim)
    print(res)