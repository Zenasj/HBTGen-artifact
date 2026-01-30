import torch.nn as nn
import random

import torch
import numpy as np

# Three classes
inputs = np.random.random((1, 3, 10, 10))
label = np.random.randint(0, 3, size=(1, 10, 10))

# Changing some labels' id to great than 3
label[0, 3, 6] = 5
label[0, 3, 8] = 4
label[0, 6, 8] = 6
label[0, 8, 2] = 5

nllloss = torch.nn.NLLLoss2d().cuda()
inputs = torch.autograd.Variable(torch.from_numpy(inputs).float()).cuda()
label = torch.autograd.Variable(torch.from_numpy(label).long()).cuda()

loss = nllloss(inputs, label)

nllloss = torch.nn.NLLLoss2d()
inputs = torch.autograd.Variable(torch.from_numpy(inputs).float())
label = torch.autograd.Variable(torch.from_numpy(label).long())

loss = nllloss(inputs, label)