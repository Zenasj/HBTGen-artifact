import torch
import torch.nn as nn

# gpu_ids: [1, 2]
model.cuda(gpu_ids[0])
model = torch.nn.DataParallel(model, device_ids=gpu_ids)