import torch
import numpy as np

for i in range(10):
    values = pickle.load(open('values-np.p','rb'))
    x = torch.from_numpy(values[0])
    y = torch.from_numpy(values[1])
    print(x.add_(-0.7,y).data.detach().numpy()[-4])

for i in range(10):
    values = pickle.load(open('values-np.p','rb'))
    x = torch.from_numpy(values[0])
    y = torch.from_numpy(values[1])
    print(x.add(-0.7,y).data.detach().numpy()[-4])

for param in net.parameters():
        param.data.add_(param.grad.data, alpha=-eta) # same as optimizer.step()

for param in net.parameters():
        param -= eta * param.grad # easiest way to implement gradient descent.