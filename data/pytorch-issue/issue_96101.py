import torch

for dtype in [torch.float32, torch.int32, torch.uint8, torch.bool]:

    torch.manual_seed(0)
    input = torch.testing.make_tensor(10, dtype=dtype, device="cpu", low=-(1 - eps), high=1 - eps)

    output = torch.atanh(input)

    if (torch.isinf(output) | torch.isnan(output)).any():
        print(f"FAIL: {dtype}")
    else:
        print(f"PASS: {dtype}")