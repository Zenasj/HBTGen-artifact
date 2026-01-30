import torch.nn as nn

import torch
layer1 = torch.nn.LSTM(10,20,batch_first=True)
layer1(torch.zeros((5,100,10,10)))[0].shape