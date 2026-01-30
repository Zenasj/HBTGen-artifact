import torch
tensor = torch.randn(23, 20, 34, 15)
tensor = tensor[:,:-3, :-2, ]
tensor = tensor.view(32, 15, 23, 17)