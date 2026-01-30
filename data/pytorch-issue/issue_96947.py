import itertools

import torch

for device, dtype in itertools.product(
    ["cpu", "cuda"],
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex32,
        torch.complex64,
        torch.complex128,
    ],
):
    torch.manual_seed(0)
    # I only used this high number of samples to make sure the other dtypes are not affected
    # On my machine 1_000 was sufficient for the check to fail for bfloat16, 
    # and 10_000 for float16 and complex32
    t = torch.rand(10_000_000, dtype=dtype, device=device)
    if dtype.is_complex:
        t = torch.view_as_real(t)

    print(f"{dtype}, {device}: {'PASS' if (t != 1).all() else 'FAIL'}")