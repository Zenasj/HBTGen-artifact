import torch
import torch.nn as nn

torch.nn.Sequential(OrderedDict([('id', torch.nn.Identity()), ('id2', torch.nn.Identity())]))

torch.nn.Sequential({'id': torch.nn.Identity(), 'id2': torch.nn.Identity()})