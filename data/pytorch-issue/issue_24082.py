import torch

curr = time.time()
torch.cuda.synchronize()
# to
torch.cuda.synchronize()
curr = time.time()