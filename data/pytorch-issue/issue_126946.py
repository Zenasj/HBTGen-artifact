import torch

# src is on host and pinned - 4GB
src = torch.randn(512, 128, 128, 128, pin_memory=True)

# async H2D
dst = src.to(device='cuda:0', non_blocking=True)

# revise the src last value
src[511,127,127,127] += 1.0

# sync the device
torch.cuda.synchronize()

# compare the result, they should not be equal
print('compare src and dst, they should not be equal, but compare result is ', torch.allclose(src, dst.cpu()))