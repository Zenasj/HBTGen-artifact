class my_loss:

    def __init__(self,weights):
      self.weights=weights

      #age loss:
      self.L1=nn.SmoothL1Loss(reduction='mean',beta=0.05)

      #gender loss:
      self.CE=nn.CrossEntropyLoss(reduction='mean')

    def __call__(self,output,target):
      loss = self.weights[0]*self.L1(output[:,0],target[:,0])+self.weights[1]*self.CE(output[:,1:3],target[:,1])

      return loss

#testing:
it=iter(dataloaders_dict['train'])
X,y=it.next()

criterion = my_loss(weights=(1,1))

out=model_ft(X)
loss=criterion(out,y)
print(y.dtype)
print(X.dtype)
print(out.dtype)
print(loss.dtype)

loss.backward()

import torch
import torch.nn as nn
out = torch.randn(3, 4, requires_grad=True)
y = torch.randint(0, 10, (3, 4), dtype=torch.int64)
L1=nn.SmoothL1Loss(reduction='mean',beta=0.05)
loss = L1(out, y)
loss.backward()