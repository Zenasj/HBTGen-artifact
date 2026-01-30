import torch

@torch.compile()
def test_mm_padding(inp, x, y):
    return torch.ops.aten.addmm(inp, x, y)

@torch.compile()
def test_mm_padding2(inp, x, y):
    return torch.ops.aten.addmm(inp, x, y, alpha=1.0, beta=1.0)

inps = [torch.rand([20]), torch.rand([20, 20]), torch.rand([20, 20])]
inps = [x.to(device="cuda") for x in inps]

print("Running first")
test_mm_padding(*inps)

print("Running second")
test_mm_padding2(*inps)