import numpy as np
import torch
from torch.autograd import Variable

x = torch.from_numpy(np.array([1.,2.,3.]))
x = Variable(x, requires_grad=True)
y = x * 2
gradients = torch.FloatTensor([1, 1, 1])
y.backward(gradients)

import numpy as np
import torch
from torch.autograd import Variable

x = torch.from_numpy(np.array([1.,2.,3.]))
x = Variable(x, requires_grad=True)
y = x * x
gradients = torch.FloatTensor([1, 1, 1])
y.backward(gradients)