import torch
import torch.nn as nn

torch.backends.cudnn.enabled = True

model = nn.Sequential(nn.Linear(10, 10), nn.LSTM(10, 1))
model.cuda()

x = torch.randn(20, 10, dtype=torch.float32).cuda()
with torch.autocast(device_type="cuda", dtype=torch.float16):
    y, _ = model(x)