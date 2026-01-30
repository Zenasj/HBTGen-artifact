import torch 
import torch.nn as nn 
import torch.optim as optim
import math
torch.manual_seed(1)
class A(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.9, affine=True):
        super(A, self).__init__()
        ks = 5
        self.a = nn.Conv1d(1, 1, kernel_size=ks, padding=(ks-1) // 2, bias=False) 
        self.b = nn.Conv1d(1, 1, kernel_size=ks, padding=(ks-1) // 2, bias=False) 
        self.bn = nn.BatchNorm2d(num_features,affine = False)
    def forward(self, x):
        out = self.bn(x)
        return out

class NetA(nn.Module):
  def __init__(self):
    super(NetA, self).__init__()

    self.input = torch.nn.parameter.Parameter(100*torch.arange(50.).reshape(1,2,5,5))

    self.bn1 = A(2,affine=False)
    self.bn2 = A(2,affine=False)
    self.conv1 = nn.Conv2d(2,2,2,1,0)
    self.conv2 = nn.Conv2d(2,2,2,1,0)
    torch.nn.init.constant_(self.conv1.weight, 2)
    torch.nn.init.constant_(self.conv2.weight,2)
  def forward(self):
    x = self.input
    x = self.bn1(x)
    x = self.conv1(x)

class NetB(nn.Module):
  def __init__(self):
    super(NetB, self).__init__()

    self.input = torch.nn.parameter.Parameter(100*torch.arange(50.).reshape(1,2,5,5))

    self.bn1 = A(2,affine=False)
    # self.bn2 = A(2,affine=False)
    self.conv1 = nn.Conv2d(2,2,2,1,0)
    self.conv2 = nn.Conv2d(2,2,2,1,0)
    torch.nn.init.constant_(self.conv1.weight, 2)
    torch.nn.init.constant_(self.conv2.weight,2)
  def forward(self):
    x = self.input
    x = self.bn1(x)
    x = self.conv1(x)

model = NetA()#model = NetB() if you want to test NetB()

model.train()
# label = torch.reshape(torch.range(1,32.),[1,2,4,4])
optimizer = optim.SGD(model.parameters(), lr=1,)
print('input')
print(model.input[0,0,0,:])


for i in range(20):
  y = model()
  loss = torch.sum(y)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

model.eval()
y = model()
print(model)
print('test')
print(y[0,0,0,:])