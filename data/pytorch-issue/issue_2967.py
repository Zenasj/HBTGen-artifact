import torch
import numpy as np

torch.from_numpy(np.array([1,2,3])).type(u'torch.DoubleTensor')