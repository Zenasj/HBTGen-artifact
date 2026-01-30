import torch

for i in range(10000):
    jitter = torch.randn((32, 1000), device="mps")
    if torch.any(torch.isnan(jitter)):
        print(f"NaN generated at iter: {i}")
        break