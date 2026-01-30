import torch.nn as nn

import torch
import torch.nn.functional as F

for func in [F.relu, F.leaky_relu, F.gelu, F.mish, lambda x: x]:
    for device in ['mps', 'cpu']:
        x1 = torch.tensor(3.0).to(device)  # This will NOT work for leakyRELU, GELU and Mish
        #x1 = torch.Tensor([3.0, -3.1]).to(device)  # This will NOT work for leakyRELU, GELU and Mish
        #x1 = torch.Tensor([[3.0, -3.1], [2.1, 3.4]]).to(device)  # This will work for all the cases
        x1.requires_grad = True
        y1 = func(x1).sum()
        y1.backward()
        print("Gradient on " + device + ":")
        print(x1.grad)
    print('----')