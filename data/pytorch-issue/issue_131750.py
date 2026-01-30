import torch

device = "mps"
rand_tensor = torch.randn(3, 4).to(device)
weights = torch.ones(4).to(device)
bias = torch.zeros(4).to(device)

_a_mps = torch.layer_norm(rand_tensor[0:1], (4,), weights, bias, 1e-5, False)
_b_mps = torch.layer_norm(rand_tensor[1:3], (4,), weights, bias, 1e-5, False)
_b_mps_clone = torch.layer_norm(rand_tensor[1:3].clone(), (4,), weights, bias, 1e-5, False)
_c_mps = torch.layer_norm(rand_tensor, (4,), weights, bias, 1e-5, False)
# should be 0, 0, 4 (on CPU and CUDA)
# but on MPS, it will be 4, 4, 4
print(torch.eq(_a_mps, _b_mps).sum(), torch.eq(_c_mps[0:1], _b_mps).sum(), torch.eq(_c_mps[1:2], _b_mps).sum())
# print 0 matches
print(torch.eq(_b_mps, _b_mps_clone).sum())

device = "cpu"
rand_tensor = rand_tensor.to(device)
weights = weights.to(device)
bias = bias.to(device)

_a_cpu = torch.layer_norm(rand_tensor[0:1], (4,), weights, bias, 1e-5, False)
_b_cpu = torch.layer_norm(rand_tensor[1:3], (4,), weights, bias, 1e-5, False)
_b_cpu_clone = torch.layer_norm(rand_tensor[1:3].clone(), (4,), weights, bias, 1e-5, False)
_c_cpu = torch.layer_norm(rand_tensor, (4,), weights, bias, 1e-5, False)
# print 0, 0, 4
print(torch.eq(_a_cpu, _b_cpu).sum(), torch.eq(_c_cpu[0:1], _b_cpu).sum(), torch.eq(_c_cpu[1:2], _b_cpu).sum())
# print 8 matches (which is all matched)
print(torch.eq(_b_cpu, _b_cpu_clone).sum())

# _a_mps equals _a_cpu
# _b_mps NOT equals _b_cpu
# _c_mps equals _c_cpu