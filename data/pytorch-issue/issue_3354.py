import torch.nn as nn
import random

import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import plotnine as p9

num_sample = 1000
num_feat = 10
num_out = 10

np.random.seed(1923)
torch.manual_seed(1923)

X = np.random.normal(0, 0.5, (num_sample, num_feat)).astype(np.float32)
W = np.random.normal(0, 0.5, (num_feat, num_out)).astype(np.float32)
b = np.random.normal(0, 0.5, (1, num_out)).astype(np.float32)
y_mean = np.dot(X, W) + b

Y = np.random.normal(y_mean, 0.5)

model = torch.nn.Sequential(torch.nn.Linear(num_feat, num_out))
loss = torch.nn.MSELoss()

optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

# dataset
full_data = TensorDataset(torch.from_numpy(X).float(),
                          torch.from_numpy(Y).float())

##### train with shuffling

print('\n##  With shuffle\n')
loader = DataLoader(full_data, batch_size=16, shuffle=True)
hist1 = []

# over epochs
for i in range(300):

    # over batches
    for j, (inp, target) in enumerate(loader):
        inp_var = Variable(inp)
        target_var = Variable(target)

        res = model(inp_var)
        l = loss(res, target_var)

        if i % 30 == 0 and j == 0:
            print('Epoch:', i, ' loss: ', l.data[0])

        if j == 0:
            hist1.append(l.data[0])

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

##### now without shuffling
model = torch.nn.Sequential(torch.nn.Linear(num_feat, num_out))
loss = torch.nn.MSELoss()

optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

print('\n##  Without shuffle\n')
loader = DataLoader(full_data, batch_size=16, shuffle=False)
hist2 = []

# over epochs
for i in range(300):

    # over batches
    for j, (inp, target) in enumerate(loader):
        inp_var = Variable(inp)
        target_var = Variable(target)

        res = model(inp_var)
        l = loss(res, target_var)

        if i % 30 == 0 and j == 0:
            print('Epoch:', i, ' loss: ', l.data[0])
        if j == 0:
            hist2.append(l.data[0])

        optimizer.zero_grad()
        l.backward()
        optimizer.step()


(p9.qplot(range(len(hist1)), hist1, xlab=' ', ylab=' ', geom='path', color='"with shuffle"') +
    p9.geom_path(p9.aes(x=range(len(hist2)), y=hist2, color='"w/o shuffle"')) +
    p9.theme_minimal()+
    p9.labs(color='Loss', x='Epoch')).save('shuffle.png')

import numpy as np
import pandas as pd
import plotnine as p9

import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, Dataset

np.random.seed(555)
torch.manual_seed(555)

num_sample = 300
num_feat = 10
num_out = 10

batch_size = 32

X = np.random.normal(0, 0.5, (num_sample, num_feat)).astype(np.float32)
W = np.random.normal(0, 0.5, (num_feat, num_out)).astype(np.float32)
b = np.random.normal(0, 0.5, (1, num_out)).astype(np.float32)
Y = np.exp(np.dot(X, W) + b)

#### training with shuffle

ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())

# train torch models with and without shuffling
model1 = torch.nn.Linear(num_feat, num_out)
opt1 = torch.optim.RMSprop(model1.parameters(), lr=0.01)
loss1 = torch.nn.MSELoss()

train_hist_shuf = []

for epoch in range(300):

    train_batch_losses = []

    # shuffle dataset
    idx = torch.randperm(len(ds))
    ds = TensorDataset(*ds[idx])

    for batch in range(int(np.ceil(len(ds)/batch_size))):
        x, y = ds[batch*batch_size:(batch+1)*batch_size]

        x_var, y_var = Variable(x), Variable(y)
        pred = model1(x_var)
        l = loss1(pred, y_var)
        train_batch_losses.append(l.data[0])

        opt1.zero_grad()
        l.backward()
        opt1.step()

    # save mean of all batch errors within the epoch
    train_hist_shuf.append(np.array(train_batch_losses).mean())

#### now no shuffling

ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())

# train torch models with and without shuffling
model2 = torch.nn.Linear(num_feat, num_out)
opt2 = torch.optim.RMSprop(model2.parameters(), lr=0.01)
loss2 = torch.nn.MSELoss()

train_hist_noshuf = []

for epoch in range(300):
    train_batch_losses = []

    for batch in range(int(np.ceil(len(ds)/batch_size))):
        x, y = ds[batch*batch_size:(batch+1)*batch_size]

        x_var, y_var = Variable(x), Variable(y)
        pred = model2(x_var)
        l = loss2(pred, y_var)
        train_batch_losses.append(l.data[0])

        opt2.zero_grad()
        l.backward()
        opt2.step()

    train_hist_noshuf.append(np.array(train_batch_losses).mean())

(p9.ggplot(pd.DataFrame({'torch_train_shuf': train_hist_shuf,
                         'torch_train_noshuf': train_hist_noshuf,
                         'epochs': range(len(train_hist_shuf))}),
            p9.aes(x='epochs')) +
  p9.geom_path(p9.aes(y='torch_train_shuf', color='"loss (shuffled)"')) +
  p9.geom_path(p9.aes(y='torch_train_noshuf', color='"loss (not shuffled)"')) +
  p9.labs(color='Loss', y=' ') +
  p9.theme_minimal()).save('shuffle.png', width=6,
                           height=4, dpi=200)