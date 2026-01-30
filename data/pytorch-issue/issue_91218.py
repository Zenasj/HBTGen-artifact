import torch.nn as nn

import torch
import torch._dynamo

def fn(x):
    r = x.mT
    return torch.nn.functional.relu(r)

x = torch.rand((2, 3, 4))

torch._dynamo.optimize("eager")(fn)(x)

def fn(x):
            try:
                unsupported = x.nonexistent_tensor_attr
                x = torch.nn.utils.rnn.PackedSequence(x, unsupported) # <- this is where the failure occurs
            except:
                ...