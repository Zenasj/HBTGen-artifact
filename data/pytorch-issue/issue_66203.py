import torch.nn as nn

import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
lstm = torch.nn.LSTM(input_size=10, hidden_size=10, num_layers=1,
                            bidirectional=True, batch_first=True)

lstm =lstm.to(torch.device('cuda:1'))
data= torch.ones([1,4,10]).float().to(torch.device('cuda:1'))
lstm(data)

import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
torch.cuda.set_device(1)
lstm = torch.nn.LSTM(input_size=10, hidden_size=10, num_layers=1,
                            bidirectional=True, batch_first=True)

lstm =lstm.to(torch.device('cuda:1'))
data= torch.ones([1,4,10]).float().to(torch.device('cuda:1'))
lstm(data)