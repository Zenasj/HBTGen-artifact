import torch
import torch.nn as nn

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.dropout = nn.Dropout(p=0.2)
		self.relu = nn.ReLU()
		self.fc1 = nn.Linear(49200, 49200)
		self.fc2 = nn.Linear(49200, 49200)
		self.fc3 = nn.Linear(49200, 3)
		self.out = nn.Sequential(
			self.fc1,
			self.relu,
			self.dropout,
			self.fc1,
			self.relu,
			self.dropout,
	 		self.fc3
			)
	def forward(self, premise, hypothesis):
		return self.out(torch.cat([premise, hypothesis], 1))

net = Net().cuda()
print (net)
premise = Variable(torch.randn(64, 82, 300))
hypothesis = Variable(torch.randn(64, 82, 300))
premise = premise.cuda()
hypothesis = hypothesis.cuda()
out = net(premise.contiguous().view(64,-1), hypothesis.contiguous().view(64,-1))
print(out)

def l2_reg(mdl):
        l2_reg = None
        for W in mdl.parameters():
                if W.ndimension() < 2:
                        continue
                else:   
                        if l2_reg is None:
                                l2_reg = (torch.max(torch.abs(W)))**2
                        else:   
                                l2_reg = l2_reg + (torch.max(torch.abs(W)))**2
      
        return l2_reg