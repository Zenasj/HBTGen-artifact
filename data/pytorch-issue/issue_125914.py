import torch

@torch.compile(fullgraph=True, backend="eager")
def cf(x):
    u0, u1 = x.tolist()
    torch._check_is_size(u0)
    torch._check_is_size(u1)
    torch._check(u0 + u1 == 20)
    if guard_size_oblivious(torch.sym_max(1, u0 + u1) == 20):
        return torch.tensor(True)
    else:
        return torch.tensor(False)

@run_test
def test_symmax():
    assert cf(torch.tensor([10, 10])).item()

max

guard_size_oblivious