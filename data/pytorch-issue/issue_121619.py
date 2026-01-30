import torch
def add_fn(a, b, out):
    res = torch.add(a, b, out=out)
    return res

if __name__ == "__main__":
    add_fn = torch.compile(add_fn)
    a = 2
    b = 3
    out = torch.tensor([])
    res = add_fn(a, b, out)
    print(res)