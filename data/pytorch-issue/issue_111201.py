import torch

def fn(inputs, dims):
    res = torch.flip(inputs, dims)
    return res

if __name__ == "__main__":
    inputs = torch.randn([2, 3, 32, 32])
    dims = [3, 2, 1, 0]
    compl_fn = torch.compile(fn, dynamic=True, backend="eager")
    res = compl_fn(inputs, dims)
    print(res)