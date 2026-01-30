import torch

torch._C._cuda_clearCublasWorkspaces()
torch._dynamo.reset()
import gc
gc.collect()
torch.cuda.empty_cache()