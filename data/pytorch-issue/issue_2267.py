import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.autograd as autograd

def Variable(data, *args, **kwargs):
    if torch.cuda.is_available():
        return autograd.Variable(data.cuda(), *args, **kwargs)
    else:
        return autograd.Variable(data, *args, **kwargs)

class LSTM_MEM_LEAK(nn.Module):
    
    def __init__(self):
      
        super(LSTM_MEM_LEAK, self).__init__()
        self.h_size = 600
        self.e_size = 900
        self.l1 = nn.Linear(256, self.e_size)
        self.lstm =nn.LSTM(self.e_size, self.h_size, batch_first = True, num_layers = 2, bidirectional = 1)
        self.l2 = nn.Linear(self.h_size*2, 300)
            
    def forward(self, input):
        
        hidden = (Variable(torch.zeros(2*2, 16, self.h_size)),
                Variable(torch.zeros(2*2, 16, self.h_size))) 
        l1 = F.relu(self.l1(input.view(-1, 256)))
        lstm_out, h = self.lstm(l1.view(16, -1, self.e_size), hidden)
        l2  = F.relu(self.l2(lstm_out.contiguous().view(-1, self.h_size*2)))
        
        return l2    
    
net = LSTM_MEM_LEAK()
net.cuda()
input = Variable(torch.rand(16, 6000, 256))
print(input.requires_grad)
out = net(input)