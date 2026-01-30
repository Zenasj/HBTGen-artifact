import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import opt_einsum as oe
import math
import random

os.putenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WORLD_SIZE"] = "1"


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

setup_seed(42)

#np.random.seed(42)
#torch.manual_seed(42)
contract = oe.contract

np.set_printoptions(precision=16)
torch.set_printoptions(precision=16)


class TransposedLinear(nn.Module):
    """ Linear module on the second-to-last dimension """

    def __init__(self, d_input, d_output, bias=True):
        super().__init__()
        self.linear = nn.Linear(3072, 64)


    def forward(self, x):
        print ("==========before call linear,---------")
        print (x)
        print (x.shape)
        print ("-------------=========-------")
        result = self.linear(x)
        print ("==========after call linear,---------")
        print (result)
        print (result.shape)

        return result

# Main function
def main():
   tensor = torch.randn(3072, 3072).cuda()

   model = TransposedLinear(3072,3072,True).cuda()
   output = model (tensor)

if __name__ == "__main__":
    main()