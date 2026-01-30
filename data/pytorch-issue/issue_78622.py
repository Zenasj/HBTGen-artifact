import torch
from torchtrustncg import TrustRegion
from torchtrustncg.utils import rosenbrock, branin
from torch import linalg as LA
import sys
import os
torch.set_printoptions(precision=24)
device = torch.device('cpu')
import numpy as np
from timeit import default_timer as timer


def energy(x):
   dimen = int(x.size()[1]/2)
   N = int(x.size()[0])
   norm = LA.vector_norm(x, dim=1, keepdim=True)
   x3 =x/norm
   x1=x3[...,:dimen] 
   x2=x3[...,dimen:]
   aa=torch.matmul(x1,x1.T)
   bb=torch.matmul(x2,x2.T)
   cc=torch.matmul(x2,x1.T)
   dd=torch.matmul(x1,x2.T)
   sq=((aa+bb)**2+(cc-dd)**2)**(5/2)
   sq2=torch.triu(sq,1)
   u=1/(N**2)*torch.triu(sq2,1).sum()
   return u


gtol = 1e-5


def closure(backward=True):
    if backward:
        optimizer.zero_grad()
    loss = energy(x0)
    if backward:
        loss.backward(create_graph=True)
    return loss

for dd in range(7,101):
    for NN in range(dd+140,200):
        x0=torch.randn((NN,2*dd),requires_grad=True,device="cpu",dtype=torch.float64)
        epsilon=.001
        en_opt = []
        for m in range(200):    
            x0=torch.randn((NN,2*dd),requires_grad=True,device="cpu",dtype=torch.float64)
            optimizer = TrustRegion([x0], opt_method='cg')
            loss=energy(x0)
            start = timer()
            for l in range(30*NN):
                    loss = optimizer.step(closure)
                    if torch.norm(x0.grad).item() < gtol:
                        break
                    if torch.norm(optimizer.param_step, dim=-1).lt(gtol).all():
                        break
                    if (l + 1) % 20 == 0:
                        print(l + 1, loss,torch.norm(x0.grad).item())
            end = timer()
            elapsed = end-start
            print(elapsed)
            os.makedirs('energy-d-{0}'.format(dd),exist_ok=True)
            os.makedirs('energy-d-{0}/energy-d-{0}-n-{1}'.format(dd,NN),exist_ok=True)
            np.savetxt('energy-d-{0}/energy-d-{0}-n-{1}/energy-d-{0}-n-{1}-energy-{2}-trial-{3}.txt'.format(dd,NN,loss,m),x0.cpu().detach().numpy(),delimiter=' ', newline='\n')
            print('---------Stage 0 Complete----------')
            print('-------------------------------------------------------')
            print(f'--Initialization--{m+1}--Complete-------')
            print('-------------------------------------------------------')