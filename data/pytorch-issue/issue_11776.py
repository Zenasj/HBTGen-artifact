import torch
import numpy as np

# This is the example from ONNX: gather_0
x = np.array([ [1.0, 1.2], [2.3, 3.4], [4.5, 5.7]])
ind_ = np.array([ [0, 1], [1, 2] ])
dim_ = 0 
# Expected output = [ [ [1.0, 1.2], [2.3, 3.4], ], [ [2.3, 3.4], [4.5, 5.7], ], ] 
y2= np.take(x,ind_,axis=dim_)

x_torch = torch.tensor(x)
ind_torch = torch.tensor(ind_)
y3 = x_torch.gather(dim_, ind_torch)

print("Input =\n", x, "\nOutput from ONNX =\n",y2, "\nOutput from PyTorch =\n",y3)