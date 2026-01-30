import torch

_100M = 100 * 1024 ** 2
r = torch.randn(_100M, dtype=torch.float32, device='cuda')
d = torch.randn(_100M, dtype=torch.float64, device='cuda')
torch.cuda.synchronize()
torch.cuda.profiler.start()
r.add_(d)
torch.cuda.profiler.stop()
torch.cuda.synchronize()