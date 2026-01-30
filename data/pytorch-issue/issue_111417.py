import torch

def transpose_inplace_mul(x):
    y = x.t()
    y.mul_(2)
    return y

x = torch.arange(6, dtype=torch.float32).reshape([2, 3])
ref_res = transpose_inplace_mul(x)
print('ref_cpu_x.stride: ', x.stride())
print('ref_cpu_x: ', x.cpu())
print('ref_cpu_res.stride: ', ref_res.stride())
print('ref_cpu_res: ', ref_res.cpu())

x = torch.arange(6, dtype=torch.float32).reshape([2, 3]).to(device='hpu')
compiled_fn = torch.compile(transpose_inplace_mul, backend='aot_hpu_training_backend')
res = compiled_fn(x)
print('hpu_x.strdides:', x.stride())
print('hpu_x: ', x.cpu())
print('hpu_res.strdides:', res.stride())
print('hpu_res: ', res.cpu())