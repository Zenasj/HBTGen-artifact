import torch
import torchvision.models as models
from torch.autograd import Variable

device = 'cuda'
input = torch.rand(16,3,224,224)
input = Variable(input)
input = input.to(device)
model = models.alexnet().to(device)
model(input)
print('done')