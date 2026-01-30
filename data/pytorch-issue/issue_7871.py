import torch.nn as nn

import torch
import onnx
import io

rnn_types = [torch.nn.RNN, torch.nn.LSTM, torch.nn.GRU]

for rnn_type in rnn_types:
    f = io.BytesIO()
    model = rnn_type(input_size=4, hidden_size=2, num_layers=1, bias=False)
    try:
        torch.onnx.export(model, torch.rand(2, 3, 4), f)
    except ValueError:
        print(rnn_type.__name__, "failed to export")