import torch
cur_xpu_device = torch.device("xpu")
torch.xpu.get_device_properties(cur_xpu_device)

import torch
print(torch.xpu.device_count())

import torch
cur_xpu_device = torch.device("xpu:0")
torch.xpu.get_device_properties(cur_xpu_device)

import torch
cur_xpu_device = torch.device("xpu:1")
torch.xpu.get_device_properties(cur_xpu_device)