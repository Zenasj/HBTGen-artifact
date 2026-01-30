import torch
import torch.nn as nn

with torch.cuda.device(1):
    model.load_state_dict(state)

model = torch.nn.DataParallel(model, device_ids=[1]) # It assigns memory on GPU 1.
with torch.cuda.device(1):
    model.load_state_dict(state) # Here, you can see the memory leak on GPU 0!

s = torch.load('chpt_0', map_location={'cuda:0':'cuda:1'})
m.load_state_dict(s)