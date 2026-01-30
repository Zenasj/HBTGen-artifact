import torch
import torch.nn as nn

model_opt = torch.nn.DataParallel(model, device_ids=device_ids)
model_opt = torch.compile(model_opt)
...