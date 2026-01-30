import torch
import torch.nn as nn
import numpy as np

class simpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
    
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hs):
        
        # forward pass thru RNN layer
        rnn_out, hidden = self.rnn(x, hs)
        
        # reshape RNN output for input to FC layer
        rnn_out = rnn_out.view(-1, self.hidden_size)
        
        # forward pass thru FC layer
        output = self.fc(rnn_out)
        
        # return prediction and updated hidden state
        return output, hidden

input_size  = 1
hidden_size = 10
num_layers  = 2
output_size = 1

test_model = simpleRNN(input_size, hidden_size, num_layers, output_size)

# Data gen 1: 
seq_len = 20

time = np.linspace(0, np.pi, seq_len + 1)
data = np.sin(time)

x = data[:-1]
y = data[1:]

test_input_1 = x.reshape((seq_len, 1)) 
test_input_1 = torch.tensor(test_input_1).unsqueeze(0) 

print(time.dtype)
print(data.dtype)
print(x.dtype)
print(test_input_1.dtype)

# Data gen 2:
seq_len = 20

time = np.linspace(0, np.pi, seq_len)
data = np.sin(time)

test_input_2 = data.reshape((seq_len, 1))
test_input_2 = torch.Tensor(test_input_2).unsqueeze(0) # give it a batch_size of 1 as first dimension

print(time.dtype)
print(data.dtype)
print(test_input_2.dtype)

test_model(test_input_2, hs=None)
test_model(test_input_1, hs=None)

test_model = simpleRNN(input_size, hidden_size, num_layers, output_size).double()