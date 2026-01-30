import random

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# backend = 'x86'
# backend = 'onednn'
backend = 'qnnpack'
torch.backends.quantized.engine = backend

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------- Generate the spiral dataset -------------
n = 10000     # Number of sampels in one class
t = np.random.rand(n*2) * 10.0
noise = np.random.randn(n*2,2) * np.sqrt(0.003)

x = 0.1*t*np.cos(t) + noise[:,0]
x[n:] *= -1
y = 0.1*t*np.sin(t) + noise[:,1]
y[n:] *= -1
data = np.stack((x, y), axis=1)
target = np.hstack( (np.ones(n), np.zeros(n)) ) # Class code

# Randomly shuffle the data
idx = np.arange(len(x))
np.random.shuffle(idx)
data, target = data[idx,:], target[idx]

dataTrain = torch.tensor(data, dtype=torch.float32, device=dev)
targetTrain = torch.tensor(target, dtype=torch.float32, device=dev).unsqueeze(1)

del n,idx,x,y

# ------------ Define the neural network ---------------
class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.layers = nn.ModuleList([
            nn.Linear(n_feature, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output)])
        self.dequant = torch.ao.quantization.DeQuantStub()
                    
    def forward(self, x):
        out = self.quant(x)
        for layer in self.layers:
            out = layer(out)
        out = self.dequant(out)
        return out

# -------------- Prepare for training -----------------
network = Net(n_feature=data.shape[1], n_hidden=32, n_output=1).to(dev)

network.eval()
network.qconfig = torch.quantization.get_default_qat_qconfig(backend)
network_fused = torch.quantization.fuse_modules(network, [['layers.0', 'layers.1']])
network_fp32_prepared = torch.quantization.prepare_qat(network_fused.train())
network_fp32_prepared.train()

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(network_fp32_prepared.parameters(), lr = 0.01)

# --------------- Train the model ------------------
max_episode = 1000

for i in range(max_episode):
    logitsTrain  = network_fp32_prepared(dataTrain)
    lossTrain = criterion(logitsTrain, targetTrain)
    optimizer.zero_grad()
    lossTrain.backward()
    optimizer.step()
    
    if i%100 == 0 or i==max_episode-1:
        print('epoch {}, loss_train={:.5f}'.format(i, lossTrain.item())) 

network_fp32_prepared.eval()
network_quant = torch.quantization.convert(network_fp32_prepared.cpu(), inplace=False)

# ----------------- Compare the results -----------------
plt.figure()
plt.plot(network_fp32_prepared(dataTrain.cpu()).detach().cpu().numpy(), network_quant(dataTrain.cpu()).numpy(), marker='.', linestyle='')
plt.title(backend)
plt.xlabel('Fake quantization output')
plt.ylabel('Real quantization output')
plt.show()