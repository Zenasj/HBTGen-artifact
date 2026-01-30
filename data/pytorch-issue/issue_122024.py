import torch

# generate args
torch.manual_seed(-123975)

# target_op
op = torch.ops.aten.max_pool2d_with_indices_backward.default

d_y = torch.randn(size=[256, 64, 56, 56], dtype=torch.float32)
d_x = torch.randn(size=[256, 64, 112, 112], dtype=torch.float32)
indices = torch.ones(size=[256, 64, 56, 56], dtype=torch.int64)

# cpu version
cpu_result = op(d_y, d_x, [3, 3], [2, 2], [1, 1], [1, 1], False, indices)
# gpu version
gpu_result = op(d_y.cuda(), d_x.cuda(), [3, 3], [2, 2], [1, 1], [1, 1], False, indices.cuda())

print(cpu_result.max())
print(gpu_result.max())