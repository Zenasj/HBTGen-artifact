import torch
import torch.nn as nn
from torch.func import jacrev, functional_call

device = "cpu"

from torch.func import jacrev, functional_call

class LSTMModel(nn.Module):
    def __init__(self, input_size, lstm_units, dense_units):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, lstm_units, batch_first=True)
        self.fc1 = nn.Linear(lstm_units, dense_units)
        self.fc2 = nn.Linear(dense_units, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = torch.tanh(lstm_out) 
        fc1_out = torch.tanh(self.fc1(lstm_out))
        output = self.fc2(fc1_out)
        return output

input_size = 2
lstm_units = 32
dense_units = 16

model = LSTMModel(input_size, lstm_units, dense_units)

inputs = torch.randn(5, 100, 2)

model.to(device); inputs = inputs.to(device)

params = dict(model.named_parameters())
# jacrev computes jacobians of argnums=0 by default.
# We set it to 1 to compute jacobians of params
jacobians = jacrev(functional_call, argnums=2)(model, params, (inputs,))