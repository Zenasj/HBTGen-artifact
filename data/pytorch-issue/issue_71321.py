import torch

torch.manual_seed(0)

# Generate random complex tensor.
X1 = torch.randn((1, 100, 50, 8), dtype=torch.complex64)  # (B, H, W, C)

# 1. Take the centered 2D ifft along dim=(1,2).
x1 = torch.fft.ifft2(X1, norm="ortho", dim=(1, 2))

# 2. Permute X1 and take centered 2D ifft.
# This fails - the error is non-zero
X2 = X1.permute((0, 3, 1, 2)).contiguous()  # (B, C, H, W)
x2 = torch.fft.ifft2(X2, norm="ortho", dim=(-2, -1))
x2 = x2.permute(0, 2, 3, 1)  # (B, H, W, C)

print("x1-x2 error: ", torch.abs(x1 - x2).sum())

def runner(norm="ortho", dtype=torch.complex64):
  # This is a copy of the code above
  torch.manual_seed(0)

  X1 = torch.randn((1, 100, 50, 8), dtype=dtype)  # (B, H, W, C)

  # 1. Take the centered 2D ifft along dim=(1,2).
  x1 = torch.fft.ifft2(X1, norm=norm, dim=(1, 2))

  # 2. Permute X1 and take centered 2D ifft.
  # This fails - the error is non-zero
  X2 = X1.permute((0, 3, 1, 2)).contiguous()  # (B, C, H, W)
  x2 = torch.fft.ifft2(X2, norm=norm, dim=(-2, -1))
  x2 = x2.permute(0, 2, 3, 1)  # (B, H, W, C)

  print(f"norm={norm}, dtype={dtype} error: ", torch.abs(x1 - x2).sum())


for dtype in [torch.complex64, torch.complex128]:
  for norm in ["forward", "ortho", "backward"]:
    runner(norm, dtype)