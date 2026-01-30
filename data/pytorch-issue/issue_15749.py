import torch.nn as nn

3
import torch

# Constants
size = 16
batch_size = 4
seq_len = 8
device = torch.device('cuda')
input_ = torch.randn(seq_len, batch_size, size).to(device)
hidden = torch.randn(1, batch_size, size).to(device)

gru = torch.nn.GRU(size, size).to(device)

# Update weight with a `torch.tensor`
# NOTE: Similar weight update as torch.nn.utils.weight_nrom
data = gru.weight_hh_l0.data
del gru._parameters['weight_hh_l0']
setattr(gru, 'weight_hh_l0', torch.tensor(data))

# Optional call to resolve parameter shapes
gru.flatten_parameters()

# Run forward pass
_, output = gru(input_, hidden)