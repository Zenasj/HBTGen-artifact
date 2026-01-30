import torch.nn as nn

import torch 
import numpy as np 

x = np.array([3.0000, 3.00001]).astype(np.float32)

torch_x = torch.Tensor(x)
torch_x.requires_grad = True
torch_hsigmoid = torch.nn.Hardsigmoid(inplace=True)

torch_x2 = torch_x + 0 # Do no operations
torch_out = torch_hsigmoid(torch_x2)
print("Torch out is: ", torch_out)

torch_out = torch.mean(torch_out)
print("Torch out is: ", torch_out)
torch_out.backward()
print("Torch grad is: ", torch_x.grad)