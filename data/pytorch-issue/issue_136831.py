import torch

tensor = torch.randn(10, device=torch.device('mkldnn'))
tensor = torch.empty(10, device=torch.device('mkldnn'))  # will also trigger
tensor = torch.ones(10, dtype=torch.float32, device=torch.device('mkldnn'))  # will also trigger
torch.zeros((2, -3), dtype=torch.float32, device='mkldnn')  # will also trigger