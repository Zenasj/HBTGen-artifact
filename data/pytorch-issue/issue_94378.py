import torch
import torch._dynamo as torchdynamo
import logging

torchdynamo.config.log_level = logging.INFO
torchdynamo.config.output_code = True


@torchdynamo.optimize("eager")
def toy_example(ta, tb):
    if ta.sum() < 0:
        return ta + 3 * tb
    else:
        return 3 * ta + tb
    
x = torch.randn(4, 4)
y = torch.randn(4, 4)
 
toy_example(x, y)