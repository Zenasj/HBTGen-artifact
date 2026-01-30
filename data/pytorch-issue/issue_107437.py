import torch

##### Setup Precedes
...

# torch.compile call
pipe = torch.compile(pipe, backend="eager")

##### Inference Follows
...