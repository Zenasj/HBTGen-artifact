import torch.nn as nn

import torch

def channel_shuffle_fn(input):
  channel_shuffle = torch.nn.ChannelShuffle(2)
  return channel_shuffle(input)

comp_model = torch.compile(channel_shuffle_fn, fullgraph=True)
input = torch.randn(1, 4, 4, 2)
output = comp_model(input)

input = torch.randn(1, 4, 2, 2)
output = comp_model(input)