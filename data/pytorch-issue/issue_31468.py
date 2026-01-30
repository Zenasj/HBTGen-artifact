import torch
import torch.nn as nn

for param in quantized_model.rnn._all_weight_values:
  print(torch.ops.quantized.linear_unpack(param.param))