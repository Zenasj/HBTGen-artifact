import torch

import numpy as np, torch, time, math, os, sys, cv2
a = torch.randn(10000, 10000)
a1 = a.to(dtype=torch.float64, device='cuda')

t=time.time(); b=torch.matmul(a1,a1); torch.cuda.synchronize(); time.time()-t