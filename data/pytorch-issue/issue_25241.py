import torch

torch.manual_seed(1234)

mat_a = torch.rand(65536, 8, 64).cuda()
mat_b = torch.rand(65536, 64, 8).cuda()

mat_a_half = mat_a.half()
mat_b_half = mat_b.half()

res = torch.bmm(mat_a, mat_b)
res_half = torch.bmm(mat_a_half, mat_b_half)

print(res[65534, 0])
print(res_half[65534, 0])
print("-" * 80)
print(res[65535, 0])
print(res_half[65535, 0])

assert torch.allclose(res, res_half.float(), atol=1e-1)