import numpy as np
import torch
t = torch.FloatTensor([np.nan])
t.clamp(-10, 10)

t.cuda().clamp(-10, 10)