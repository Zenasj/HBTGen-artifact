import torch.nn as nn

from torch import nn
import torch
import numpy as np
class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.a = nn.Parameter(torch.sparse_coo_tensor((np.array([0,1]),np.array([0,1])), np.array([1.0,2.0]), size=[2,2],
                                                dtype=torch.float32), requires_grad=False)

m = Test()
m.load_state_dict(m.state_dict())