import torch
import torch.nn as nn
import traceback

args =  {'input': torch.rand([]), 'out': torch.rand([2, 2, 8, 2, 6, 6, 8, 1, 7], dtype=torch.float64), 'tensor1': torch.rand([6]), 'tensor2': torch.rand([1]), 'value': 2}
torch.addcdiv(**args)