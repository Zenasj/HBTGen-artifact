import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda:0'
embed_dim = 2
hidden_dim = 2
B = 4
T = 1

data = torch.randn(B,T,embed_dim, device=device)
h0 = torch.randn(B, embed_dim, device=device)
h0 = (h0, h0)

w_hh = torch.randn(4*hidden_dim, hidden_dim, device=device, requires_grad=True)
w_ih = torch.randn(4*embed_dim, hidden_dim, device=device, requires_grad=True)
b_hh = torch.randn(4*hidden_dim, device=device, requires_grad=True)
b_ih = torch.randn(4*embed_dim, device=device, requires_grad=True)

hiddens = h0
input = data
T = input.size(1) # check if time is actually the first dim
for t in range(T): 
    hiddens = torch.lstm_cell(input[:, t, :], hiddens, w_ih, w_hh, b_ih, b_hh)

h, c = hiddens
loss = c.sum() #+0*h.sum()
loss.backward()