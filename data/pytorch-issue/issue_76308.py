import torch

numbers = torch.tensor([0.235434, 0.345345, 0.45645])
rounded_quats = torch.round(numbers, decimals=2)