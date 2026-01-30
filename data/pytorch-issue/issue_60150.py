nvidia-smi

torch.cuda.empty_cache()

import torch
import time

temp = torch.ones([4000, 4000], device='cuda')
time.sleep(5)
del temp
torch.cuda.empty_cache()
time.sleep(1000)

temp = torch.ones([4000, 4000], device='cuda')