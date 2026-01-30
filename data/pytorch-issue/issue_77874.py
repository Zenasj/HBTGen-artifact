import torch
import craft
net = craft.CRAFT()
state_dict = torch.load('craft.pth', map_location=torch.device('mps'))

from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove module.
    new_state_dict[name] = v
dict = net.load_state_dict(new_state_dict)
print (dict)