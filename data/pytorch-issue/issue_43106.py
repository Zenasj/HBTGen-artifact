import torch
import torchvision
import os
torch.save(torchvision.models.resnet50(pretrained=True).state_dict(), 'resnet50.pt')
state_dict = torch.hub.load_state_dict_from_url(f'file://{os.getcwd()}/resnet50.pt')

torch.save(torchvision.models.resnet50(pretrained=True).state_dict(), 'resnet50.pt', _use_new_zipfile_serialization=False)
state_dict = torch.hub.load_state_dict_from_url(f'file://{os.getcwd()}/resnet50.pt')