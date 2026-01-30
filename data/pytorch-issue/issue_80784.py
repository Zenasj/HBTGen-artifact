import torch

relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))