import torch

torch.use_deterministic_algorithms(False)
operation()
torch.use_deterministic_algorithms(True)