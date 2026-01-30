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
        self.rnn = nn.LSTM(input_size = self.n_input, hidden_size = self.hidden_size,num_layers = self.n_layers)
        self.layer_out = nn.Linear(self.hidden_size,1)

        self.time_trace = torch.func.vmap(torch.trace)

    def forward(self,x) : 
        self.jac = torch.func.vmap(torch.func.jacrev(self.RNN_forward))
        der_out = torch.diagonal(self.jac(x),dim1=1,dim2=3)[:,0].transpose(2,3).transpose(1,2)
        
        return der_out
    
    def RNN_forward(self,x) :
        output,_ = self.rnn(self.time_trace(x)[:,None])
        return self.layer_out(output)

if __name__ == "__main__" : 
    batch = 10
    seq_len = 12
    Net = MyNet(1,4,20)

    x = torch.rand(batch,seq_len,3,3)
    x.requires_grad = True
    y_truth = torch.rand(batch,seq_len,3,3)
    y_pred = Net(x)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(Net.parameters(), lr = 1e-3)

    optimizer.zero_grad()
    loss = criterion(y_pred,y_truth)
    loss.backward()
    optimizer.step()