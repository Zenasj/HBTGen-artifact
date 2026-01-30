import torch 
import torch.nn as nn
import torch.nn.functional as F

SPARSE_GRAD = True

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.weight = nn.Parameter(torch.rand(2, 10)) 

    def forward(self, x):
        index = torch.tensor([[7, 4, 5], [0, 1, 2]])
        w = torch.gather(self.weight, dim=1, index=index, sparse_grad=SPARSE_GRAD) 
        loss = F.linear(x, w)
        return loss.mean()

net = Net()
params = net.parameters()
optimizer = torch.optim.SGD(params, lr=1e-3, momentum=0.9) 

for i in range(10):
    x = torch.rand(32, 3) 
    loss = net(x)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print('Batch {}, Loss {}'.format(i, loss.detach().cpu().numpy()))