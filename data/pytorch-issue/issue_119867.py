import torch

torch.compile()
def f(a, b, mat1, mat2):
    bias = torch.bmm(a + 3.14, b).permute(0, 2, 1).reshape(3992, -1)
    return torch.addmm(bias, mat1, mat2)
f(
    torch.randn(3992, 20, 40).cuda(),
    torch.randn(3992, 40, 192).cuda(),
    torch.empty(3992, 1024).cuda(),
    torch.empty(1024, 3840).cuda(),
)