import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.param = nn.Parameter(torch.Tensor(3, 5))
        self.device = torch.device('cpu:0')

net = Net()
torch.save(net.state_dict(), "net.pth")  # OK
torch.save(net, "net2.pth")              # errors out