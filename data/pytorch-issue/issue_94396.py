import torch

random_values = torch.randn(224, 224, 3, device=torch.device("mps"))
permuted_clampped_random_values = torch.clamp(values.permute(2, 0, 1), min=0, max=0.5)
print(random_values.max())
print(permuted_clampped_random_values.max())