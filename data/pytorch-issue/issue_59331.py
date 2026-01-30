import torch

torch.manual_seed(1234)

mat_a = torch.rand(512, 8, 64).cuda()
mat_b = torch.rand(512, 64, 8).cuda()

mat_a_half = mat_a.half()
mat_b_half = mat_b.half()

res_list = []
for i in range(0, 512):
  res_tmp =torch.mm(mat_a_half[i], mat_b_half[i])
  res_list += [res_tmp]
res = torch.cat(res_list, dim=0).reshape(512,8,8)

res_half = torch.bmm(mat_a_half, mat_b_half)

assert torch.allclose(res, res_half)