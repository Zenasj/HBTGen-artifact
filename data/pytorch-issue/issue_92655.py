import torch
import torch._dynamo
import logging

torch._dynamo.config.log_level = logging.DEBUG
torch._dynamo.config.verbose = True

def fn(x):
    y = torch.div(4, 4, rounding_mode='trunc')
    # y = torch.tensor(1)     <- this way works well
    return x.view(y, 16)


opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
print(opt_fn(x))

FakeTensor.constant