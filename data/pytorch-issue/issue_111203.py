import torch


def fn(dim, index, source, alpha):
    res = inputs.index_add(dim, index, source, alpha=alpha)
    return res


if __name__ == "__main__":
    inputs = torch.randn([4, 16, 32, 8])
    dim = 1
    alpha = 2
    index = torch.tensor([0, 1, 2])
    source = torch.randn([4, 3, 32, 8])
    compl_fn = torch.compile(fn, dynamic=True, backend="eager")
    res = compl_fn(dim, index, source, alpha)
    print(res)