import torch.nn as nn
import random

import numpy as np 
import matplotlib.pyplot as plt 
import torch

device = torch.device('cpu')

# function to get all the params from a pytorch model
def getParams(model):
    a = list(model.parameters())
    b = [a[i].detach().cpu().numpy() for i in range(len(a))]
    c = [b[i].flatten() for i in range(len(b))]
    d = np.hstack(c)

    return d

# set up a simple model (9 params)
testModule = torch.nn.Conv2d(1, 1, kernel_size = (3, 3), bias = False, stride = 1, padding = 1).double()
torch.nn.init.normal_(testModule.weight, mean=0, std=1)
testModule = testModule.eval()

# set up a dummy input
patch = torch.from_numpy(np.random.randn(1,1,80,80).astype('double')).to(device)

# apply the model 100 times
testVals = []
testParams = []
testModuleOut = []
for ii in range(100):
    testParams.append(getParams(testModule))
    testModuleOut.append(testModule(patch).cpu().detach()[0,:,:,:].numpy())

testParams = np.stack(testParams)
testModuleOut = np.stack(testModuleOut)

# view the variation of the model parameters and the output values
plt.figure()
plt.plot(np.std(testParams,axis=0))
plt.xlabel('Parameter index')
plt.ylabel('Standard deviation over runs')

plt.figure()
plt.plot(np.std(testModuleOut,axis=0).ravel())
plt.xlabel('Output index')
plt.ylabel('Standard deviation over runs')

arr = np.zeros((100,1),dtype='double')
arr.fill(np.random.randn(1)[0])
print(np.std(arr))
# output e.g. 2.220446049250313e-16