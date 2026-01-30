import torch

with torch.device("cpu"):
    print(torch.ones((2,), dtype=torch.complex64).abs())
    

with torch.device("mps"):
    print(torch.ones((2,), dtype=torch.complex64).abs())
    print(torch.tensor([1.0 + 0.0j, 0.0 + 10.0j, 100.0 + 0.0j, 1000.0 + 0.0j]).abs())