import torch.nn as nn
import torch.nn.functional as F

import torch
beta, threshold = 13, 7
x=torch.linspace(-4,6,1000)
s=torch.nn.Softplus(beta=beta,threshold=threshold)
torch.onnx.export(s,x,'test.onx')

from torch import Tensor
from torch.nn import functional as F

class Softplus2(torch.nn.Softplus):
      def forward(self, input: Tensor) -> Tensor:
        return F.softplus(input*self.beta, 1, self.threshold)/self.beta
    
s2=Softplus2(beta=beta,threshold=threshold)

y=s(x)
y2=s2(x)
torch.testing.assert_allclose(y,y2)

torch.onnx.export(s2,x,'test2.onx')