import torch


def fn(inputs, dims):
    res = torch.var(inputs, dims)
    return res


if __name__ == "__main__":
    inputs = torch.randn([8, 3, 2, 2])
    dims = 3
    compl_fn = torch.compile(fn, dynamic=True, backend="eager")
    res = compl_fn(inputs, dims)
    print(res)