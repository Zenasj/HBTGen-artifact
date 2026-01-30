import torch
import torch.nn as nn

cpu_mesh = DeviceMesh('cpu', [0, 1])
global_x = torch.rand(256, 256)
global_w = torch.rand(256, 256)
global_b = torch.rand(256)

dist_x = distribute_tensor(global_x, cpu_mesh, [Replicate()])
dist_w = distribute_tensor(global_w, cpu_mesh, [Replicate()])
dist_b = distribute_tensor(global_b, cpu_mesh, [Replicate()])

result1 = torch.nn.functional.linear(dist_x, dist_w, dist_b) #  hits DTensor.__torch_dispatch__ as expected
result2 = custom_op(dist_x, dist_w, dist_b) # never hits DTensor.__torch_dispatch__. Directly hits the C++ implementation