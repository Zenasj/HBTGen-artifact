import torch
def func(x):
    if x.type() == "torch.hpu.FloatTensor":
        return x
    else:
        return x + 1


if __name__ == "__main__":
    compiled_func = torch.compile(func, backend="hpu_backend")
    x = torch.tensor([1.0], device="hpu")
    print(f"{func(x)=}")
    print(f"{compiled_func(x)=}")