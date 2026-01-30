import torch.nn as nn
import numpy as np

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence


# Make some sparsely populated synthetic data,
# mimicking padded time-series data
samples = 1000

input_data = torch.zeros((samples, 50, 3))
p = torch.rand((samples, 50, 3))
new_values = torch.normal(5, 3, size=(samples, 50, 3))
idx = torch.where(p < 0.05)
input_data[idx] = new_values[idx]
idx = input_data.sum(2).sum(1).nonzero().flatten()
input_data = input_data[idx]

# Set to mps
input_data = input_data.to('mps')

# Pack the data
lengths = Tensor([1 + arr.nonzero()[:, 0].max().item() - arr.nonzero()[:, 0].min().item()
                    for arr in input_data.cpu()])

packed_data = pack_padded_sequence(
            input_data,
            lengths,
            batch_first=True,
            enforce_sorted=False)

# Make our neural network
net = nn.Sequential(
    nn.LSTM(input_size=3,
            hidden_size=50)).to('mps')

output, (hn, cn) = net(packed_data)
print(output)

import random
import torch

from torch.nn.utils.rnn import pack_padded_sequence

def test_lstm(num_layers, bidirectional, batch_first, sent_len, num_sentences):
    torch.manual_seed(1234)
    random.seed(1234)
    lstm = torch.nn.LSTM(10, 1024, num_layers=num_layers, bidirectional=bidirectional, batch_first=batch_first)
    lstm.eval()

    inp = torch.randn(sent_len, num_sentences, 10)
    if batch_first:
        inp = inp.transpose(0, 1)
    # random length "sentences"
    offsets = [random.randint(2, sent_len) for _ in range(num_sentences)]
    offsets = sorted(offsets, reverse=True)
    inp = pack_padded_sequence(inp, offsets, batch_first=batch_first)

    print("num_layers {} bidirectional {} batch_first {} sent_len {} num_sentences {} inp.data.shape {}".format(num_layers, bidirectional, batch_first, sent_len, num_sentences, inp.data.shape), end="  ", flush=True)
    lstm_cpu_result = lstm(inp)[0]
    lstm_mps_result = lstm.to("mps")(inp.to("mps"))[0]
    lstm_cpu_norm = torch.linalg.norm(lstm_cpu_result.data).item()
    lstm_mps_norm = torch.linalg.norm(lstm_mps_result.data).item()
    lstm_diff_norm = torch.linalg.norm(lstm_mps_result.data.to("cpu") - lstm_cpu_result.data).item()
    print(lstm_cpu_norm, lstm_mps_norm, lstm_diff_norm)

torch.linalg.norm(torch.randn(5, 5, 5).to("mps")) # less noisy later

test_lstm(2, True, True, sent_len=50,  num_sentences=639)   # works
test_lstm(2, True, True, sent_len=50,  num_sentences=640)   # works

test_lstm(2, True, True, sent_len=100, num_sentences=639)   # works
test_lstm(2, True, True, sent_len=100, num_sentences=640)   # dies