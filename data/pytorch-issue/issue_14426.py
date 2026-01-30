import torch
import torch.nn as nn
import numpy as np

x = torch.autograd.Variable(torch.FloatTensor(np.ones((5,2))*np.expand_dims(np.linspace(-1,1,5),1)), requires_grad=True)
f = torch.tanh(x[:,0]**2*x[:,1]**2+2*x[:,0])

f.backward(torch.ones_like(f),retain_graph=True,create_graph=True)
dx=x.grad
print(dx)

x.grad.data.zero_()
dx[:,0].backward(torch.ones_like(dx[:,0]),retain_graph=True)
dx2 = x.grad
print(dx2)

x.grad.data.zero_()
dx[:,1].backward(torch.ones_like(dx[:,1]),retain_graph=True)
dx3 = x.grad
print(dx3)

#Model
x = np.linspace(-1,1,100)
y = x**4

X_pt = torch.autograd.Variable(torch.FloatTensor(np.expand_dims(x,1)),requires_grad=True)
y_pt = torch.FloatTensor(np.expand_dims(y,1))

model = torch.nn.Sequential(torch.nn.Linear(1,10),
                            torch.nn.ELU(),
                            torch.nn.Linear(10,1))
loss_fn = torch.nn.MSELoss()
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
steps=50000
for i in np.arange(steps):
    y_pred = model(X_pt)

    loss = loss_fn(y_pred, y_pt)
    if i%10000==0:
        print(i, loss.item())
  
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

#Gradient calculation
X_pt.grad.data.zero_()
y_pred.backward(torch.ones_like(y_pred),retain_graph=True,create_graph=True)
dx=X_pt.grad
X_pt.grad.data.zero_()
dx.backward(torch.ones_like(dx),retain_graph=True)
dx2 = X_pt.grad