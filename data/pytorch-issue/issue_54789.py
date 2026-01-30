import numpy as np
import torch
# these are the same data and they look the same when you load them, but they behave subtly differently
poisoned = np.frombuffer(b'\xff\xff\xff\xff', dtype=np.bool) # hex 255
clean = np.frombuffer(b'\x01\x01\x01\x01', dtype=np.bool)

torch_poisoned = torch.from_numpy(poisoned)
torch_clean = torch.from_numpy(clean)


print('poisoned == clean -->', (poisoned == clean).all())  # >>> True
print('torch_poisoned == torch_clean --> ', (torch_poisoned == torch_clean).all())  # >>> tensor(False)

print('torch_poisoned.dtype == torch_clean.dtype --> ', (torch_poisoned.dtype == torch_clean.dtype))  # >>> True
print('torch_poisoned.dtype, torch_clean.dtype --> ', torch_poisoned.dtype, torch_clean.dtype)  # >>> torch.bool, torch.bool

print('torch_clean: ',torch_clean)  # >>> tensor([True, True, True, True])
print('torch_clean.float(): ', torch_clean.float())  # >>> tensor([1., 1., 1., 1.])

print('torch_poisoned: ', torch_poisoned)  # >>> tensor([True, True, True, True])
print('torch_poisoned.float(): ', torch_poisoned.float())  # >>> tensor([255., 255., 255., 255.])

import numpy as np
import torch
from PIL import Image, ImageDraw

mask = Image.new(mode='1', size=(4, 4), color=False)
draw = ImageDraw.Draw(mask)
draw.polygon([(1, 1), (2, 1), (2, 2), (1, 2)], fill=True)

np_mask = np.array(mask)
print(np_mask)
tensor_mask = torch.from_numpy(np_mask)
print(tensor_mask) 
float_mask = tensor_mask.float()
print(float_mask)

import numpy as np
import torch

device = torch.device('cuda')
a = np.frombuffer(b'\xff\x00', dtype=bool)
print(a)                # >>> [ True False]
t = torch.from_numpy(a).to(device)
print(t)                # >>> tensor([ True, False], device='cuda:0')
print(t.float())        # >>> tensor([-1.,  0.], device='cuda:0')

a = np.frombuffer(b'\xff\x00', dtype=np.bool_)
t = torch.tensor(a, dtype=torch.float)
print(t)