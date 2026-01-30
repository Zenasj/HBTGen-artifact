import torch
import torch.nn as nn

def logsigmoid(x):
    return -torch.nn.functional.softplus(-x)