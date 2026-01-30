import torch
print(torch.__version__)    #  1.13.0.dev20220913+cpu
torch.max(input=torch.tensor([1]), other=torch.ones([1,1,1]), out=torch.ones([1,5,1,1,1]))

import torch
print(torch.__version__)    #  1.13.0.dev20220913+cpu
torch.min(input=torch.tensor([1]), other=torch.ones([1,1,1]), out=torch.ones([1,5,1,1,1]))