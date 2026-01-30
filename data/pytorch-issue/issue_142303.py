import torch.nn as nn

import torch

batch_first=False
num_layers = 1
hidden_size = 5
input_size = 2 

rnn = torch.nn.RNN(
    input_size=input_size, 
    hidden_size=hidden_size, 
    num_layers=num_layers
)

weight_ih = [rnn.weight_ih_l0.data]
bias_ih = [rnn.bias_ih_l0.data]
weight_hh = [rnn.weight_hh_l0.data]
bias_hh = [rnn.bias_hh_l0.data]

def forward(x, h_0=None):
    if batch_first:
        x = x.transpose(0, 1)
    seq_len, batch_size, _ = x.size()
    if h_0 is None:
        h_0 = torch.zeros(num_layers, batch_size, hidden_size)
    h_t_minus_1 = h_0
    h_t = h_0
    output = []
    for t in range(seq_len):
        for layer in range(num_layers):
            h_t[layer] = torch.tanh(
                x[t] @ weight_ih[layer].T
                + bias_ih[layer]
                + h_t_minus_1[layer] @ weight_hh[layer].T
                + bias_hh[layer]
            )
        output.append(h_t[-1])
        h_t_minus_1 = h_t
    output = torch.stack(output)
    if batch_first:
        output = output.transpose(0, 1)
    return output, h_t


example_input = torch.randn(3, 4, input_size)

print(rnn(example_input)[0])
print(forward(example_input)[0])