import torch
torch.set_default_tensor_type(torch.cfloat)

x = torch.randn(2,2, dtype=torch.cfloat)

tensor([[-0.0099-0.3028j, -0.5884+0.1538j],
        [-0.5426-0.9435j, -0.8310+0.4520j]])