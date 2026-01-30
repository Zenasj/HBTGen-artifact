import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BatchNormTest(nn.Module):
    def __init__(self, c, num_classes=2):
        super(BatchNormTest, self).__init__()
        self.bn = nn.BatchNorm2d(c)

    def forward(self, x):
        out = x
        out = self.bn(out)
        out = F.relu(out)
        return out

c = 100
net = BatchNormTest(c)
use_cuda = True
inputs = Variable(torch.rand(100,c,100,100), requires_grad=True)
if use_cuda:
    net.cuda()
    inputs = inputs.cuda()

T = 100
for i in range(T):
    output = net(inputs)
    loss1 = torch.sum(output)
    grad_params = torch.autograd.grad(loss1, inputs, create_graph=True)

    grad = grad_params[0]
    loss = torch.sum(grad)

    loss.backward()
    print(i)