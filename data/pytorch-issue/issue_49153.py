import torch

torch.jit.trace(model, args).graph