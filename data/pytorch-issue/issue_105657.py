import torch._dynamo.config
import logging

torch._dynamo.config.log_level = logging.INFO
torch._dynamo.config.output_code = True

import torch
import torch._dynamo

def g(x):
    return (x + x)

x = torch.tensor(12)
print(torch.compile(g)(x))