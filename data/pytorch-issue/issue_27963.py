import torch.nn as nn

import torch
from torch.nn.utils.rnn import pack_sequence

# quantized apis require a container, not a top-level module - thus sequential
m = torch.nn.Sequential(torch.nn.LSTM(2,5,num_layers=2))
x = pack_sequence([torch.rand(4,2), torch.rand(3,2), torch.rand(2,2)])
m(x) # works

qm = torch.quantization.quantize_dynamic(m, dtype=torch.qint8)
qm(x)