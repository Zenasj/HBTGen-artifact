import torch
import torch._lazy
import torch._lazy.ts_backend
torch._lazy.ts_backend.init()

torch.tensor([1], device="cpu") + torch.tensor([1], device="lazy")