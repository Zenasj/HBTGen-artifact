import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

@torch.no_grad()
def generate_inputs_grid_sampling(dtype):
    N, C, Hi, Wi = 32, 64, 16, 16
    N, Ho, Wo = N, Hi//2, Wi//2
    
    inputs = torch.randn([N, C, Hi, Wi])
    grid = torch.rand([N, Ho, Wo, 2]) * 2 - 1
    return (
        inputs.cuda().to(dtype),
        grid.cuda().to(dtype),
    )

def calc_err(x1, x2):
    err = (x1 - x2).abs()
    err_rel = err / x2.abs()
    max_abs_err = err.max()
    max_rel_err = err_rel.max()
    mean_rel_err = err_rel.mean()
    return max_abs_err.item(), max_rel_err.item(), mean_rel_err.item()

def sampling(inputs, grid):
    return F.grid_sample(inputs, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

def info_versions():
    print(f'Pytorch version: {torch.__version__}')
    print(f'CUDA version: {torch.version.cuda}')
    print(f'NCCL version: {torch.cuda.nccl.version()}')

if __name__ == '__main__':
    info_versions()

    inputs, grid = generate_inputs_grid_sampling(dtype=torch.float16)
    out = sampling(inputs, grid)

    inputs_fp64, grid_fp64 = inputs.double(), grid.double()
    out_fp64 = sampling(inputs_fp64, grid_fp64)

    max_abs_err, max_rel_err, mean_rel_err = calc_err(out, out_fp64)

    print(f'max_abs_err: {max_abs_err:.8f}; max_rel_err: {max_rel_err:.8f}; mean_rel_err: {mean_rel_err:.8f}; ')