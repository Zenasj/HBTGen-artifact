import torch

torch.experimental.deterministic = True
torch.some_operation(deterministic=False)