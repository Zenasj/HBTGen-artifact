import torch

z = torch.rand((1, 32)).requires_grad_()

repeated = z.repeat(1024, 1)
repeated = z.repeat_interleave(1024, dim=0)
repeated = z.expand(1024, 32)