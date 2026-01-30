import torch

torch.fmod(a, b) == a - a.div(b, rounding_mode="trunc") * b