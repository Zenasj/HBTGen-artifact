import torch

torch.slogdet(torch.rand(128,128).to('cuda'))  # Succeeds
torch.slogdet(torch.rand(129,129).to('cuda'))  # Fails
torch.slogdet(torch.rand(256,256).to('cuda'))  # Also fails <- isn't a power of 2 thing