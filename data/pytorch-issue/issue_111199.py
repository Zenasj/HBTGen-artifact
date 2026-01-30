import torch


def fn(inputs, other, alpha, out):
    res = torch.add(inputs, other, alpha=alpha, out=out)
    return res


if __name__ == "__main__":
    inputs = torch.randn(2, 3, 4)
    other = 1
    alpha = 2
    out = torch.empty(2, 3, 4)
    compl_fn = torch.compile(fn, dynamic=True)
    res = compl_fn(inputs, other, alpha, out)
    print(res)