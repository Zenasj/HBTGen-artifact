import torch.nn as nn

import torch
import torch.nn.functional as F
elu_input = torch.randn((1,1024,500))
alpha = float(1)
inplace = False
#print(F.elu(elu_input.to('cpu'), alpha, inplace))
#print(F.elu(elu_input.to('mps'), alpha, inplace))
print("contiguous:", elu_input.is_contiguous(), 
      ", all close:", torch.allclose(
    F.elu(elu_input.to('cpu'), alpha, inplace).cpu(),
    F.elu(elu_input.to('mps'), alpha, inplace).cpu(),
    rtol=1e-3, atol=1e-6)
)
print("")

elu_input_noncontiguous = elu_input.transpose(1, 2)
#print(F.elu(elu_input.to('cpu'), alpha, inplace))
#print(F.elu(elu_input.to('mps'), alpha, inplace))
print("contiguous:", elu_input_noncontiguous.is_contiguous(), 
      ", all close:", torch.allclose(
    F.elu(elu_input_noncontiguous.to('cpu'), alpha, inplace).cpu(),
    F.elu(elu_input_noncontiguous.to('mps'), alpha, inplace).cpu(),
    rtol=1e-3, atol=1e-6)
)
print("non-contiguous delta:", F.elu(elu_input_noncontiguous.to('cpu'), alpha, inplace).cpu() - 
    F.elu(elu_input_noncontiguous.to('mps'), alpha, inplace).cpu())