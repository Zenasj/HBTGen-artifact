import torch.nn as nn

import torch
from torch import nn
import torch.optim as optim

class MyNet(nn.Module) :
    def __init__(self,n_input,n_layers,hiddden_size):
        super(MyNet,self).__init__()
        self.n_input = n_input
        self.n_layers = n_layers
        self.hidden_size = hiddden_size

        # Layers
        self.rnn = nn.LSTM(input_size = self.n_input, hidden_size = self.hidden_size,num_layers = self.n_layers, batch_first = True)
        self.layer_out = nn.Linear(self.hidden_size,1)

    def forward(self,x) : 
        out,_ = self.rnn(x)
        out = self.layer_out(out)

        der_out = torch.autograd.grad(out,x,
                                      grad_outputs = torch.ones_like(out),
                                      retain_graph = True,
                                      create_graph = True)[0]
        return der_out

if __name__ == "__main__" : 
    batch = 10
    seq_len = 12
    input = 3
    Net = MyNet(3,3,20)

    x = torch.rand(batch,12,3)
    x.requires_grad = True
    y_truth = torch.rand(batch,12,3)
    y_pred = Net(x)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(Net.parameters(), lr = 1e-3)

    optimizer.zero_grad()
    loss = criterion(y_pred,y_truth)
    loss.backward()
    optimizer.step()