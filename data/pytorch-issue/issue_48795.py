import torch
import torchvision
model = torchvision.models.densenet161(pretrained=False)
model.eval()
torch.jit.script(model).save('densenet161.pt')

import torch
import torchvision
model = torchvision.models.inception_v3(pretrained=False)
model.eval()
torch.jit.script(model).save('inception_v3.pt')