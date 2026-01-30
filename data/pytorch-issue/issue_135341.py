import torch


def fn(tensor):
    res = torch.signbit(tensor)
    return res


if __name__ == "__main__":
    tensor = torch.randn((2, 3, 4), dtype=torch.bfloat16)
    cm_fn = torch.compile(fn)
    res = cm_fn(tensor)
    print(res)