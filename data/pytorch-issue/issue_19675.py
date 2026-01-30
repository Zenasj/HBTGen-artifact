import torch
import torch.nn as nn

input = torch.rand([60, 8, 256]) #batch size is 60, sequence size is 8, features are 256
lstm = torch.nn.LSTM(256, 512, batch_first=True)
_, (finalHiddenState, _) = lstm(input)
print(finalHiddenState.shape)
Ouput: torch.Size([1, 60, 512])

output, (hx, cx) = lstm(input)