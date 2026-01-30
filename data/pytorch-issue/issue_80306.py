3
import torch
from torch.nn import LSTM

assert torch.backends.mps.is_available()

tensor_shapes = {}

for device_type in ("mps", "cpu"):
    device = torch.device(device_type)
    lstm = LSTM(64, 128, 1, batch_first=True)
    lstm.to(device)
    input_tensor = torch.randn((2, 4, 64)).to(device)
    output, _ = lstm(input_tensor)
    tensor_shapes[device_type] = output.shape
    
print(tensor_shapes) # prints {'mps': torch.Size([4, 2, 128]), 'cpu': torch.Size([2, 4, 128])}
                     #                            ^^^^^                           ^^^^^

3
import torch
from torch.nn import LSTM

assert torch.backends.mps.is_available()

tensor_shapes = {}

for device_type in ("mps", "cpu"):
    device = torch.device(device_type)
    lstm = LSTM(64, 128, 1)
    lstm.to(device)
    input_tensor = torch.randn((2, 4, 64)).to(device)
    output, _ = lstm(input_tensor)
    tensor_shapes[device_type] = output.shape
    
print(tensor_shapes) # prints {'mps': torch.Size([2, 4, 128]), 'cpu': torch.Size([2, 4, 128])}

3

import torch
from torch.nn import GRU

assert torch.backends.mps.is_available()

tensor_shapes = {}

for device_type in ("mps", "cpu"):
    device = torch.device(device_type)
    gru = GRU(64, 128, 1, batch_first=True)
    gru.to(device)
    input_tensor = torch.randn((2,4,64)).to(device)
    output, _ = gru(input_tensor)
    tensor_shapes[device_type] = output.shape
    
print(tensor_shapes) # prints {'mps': torch.Size([2, 4, 128]), 'cpu': torch.Size([2, 4, 128])}