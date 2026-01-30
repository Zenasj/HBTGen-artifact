import torch.nn as nn

import torch
class Net(torch.nn.Module):    
    def __init__(self, dim = [1,20,1]):
        super(Net, self).__init__()
        self._net = FCN(dim[0],dim[1],dim[-1])
 
    def forward(self, u, x):
        y_tr = self._net(x)
        y_out = torch.einsum('BbO,BSbO->BSbO',u,y_tr)
        return y_out
class FCN(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(FCN, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden,bias=True) 
        self.hidden2 = torch.nn.Linear(n_hidden,  n_hidden,bias=True)
        self.hidden3 = torch.nn.Linear(n_hidden,  n_hidden,bias=True)
        self.hidden4 = torch.nn.Linear(n_hidden,  n_hidden,bias=True)
        self.predict = torch.nn.Linear(n_hidden,  n_output,bias=True)   
    def forward(self, y):
        x = self.hidden1(y)
        x = torch.sin(x)              
        x = self.hidden2(x)
        x = torch.sin(x)
        x = self.hidden3(x)
        x = torch.sin(x)
        x = self.hidden4(x)
        x = torch.sin(x)
        x = self.predict(x)                        
        return x

t = torch.linspace(0.0125,1,80).view(80,1).repeat(5,40,1,1)
t.requires_grad = True
tsize = t.size()
u = torch.rand(tsize[0],tsize[2],20)

model = Net(dim=[1,100,20])
test = model(u,t)
Grad_auto = torch.autograd.grad(test[:,0,:,0],t,retain_graph=True,grad_outputs=torch.ones_like(test[:,0,:,0]))[0][0,0,:,0]
Grad_Num = torch.gradient(test[:,0,:,0],spacing=0.0125, edge_order=2, dim=1)[0][0,:]

model = Net(dim=[1,100,20])
test = model._Net(t)
Grad_auto = torch.autograd.grad(test[:,0,:,0],t,retain_graph=True,grad_outputs=torch.ones_like(test[:,0,:,0]))[0][0,0,:,0]
Grad_Num = torch.gradient(test[:,0,:,0],spacing=0.0125, edge_order=2, dim=1)[0][0,:]

model = Net(dim=[1,100,20])
test = model(u,t)
Grad_auto = torch.autograd.grad(test[:,0,:,0],t,retain_graph=True,grad_outputs=torch.ones_like(test[:,0,:,0]))[0][0,0,:,0]
Grad_Num = torch.gradient(test[:,0,:,0],spacing=0.0125, edge_order=2, dim=1)[0][0,:]

t = torch.linspace(0.0125,1,80).view(80,1).repeat(5,40,1,1)