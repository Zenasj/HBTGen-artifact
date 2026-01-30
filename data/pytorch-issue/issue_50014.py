import torch
import torchvision.models as models

print('start')
resnet18 = models.resnet18(pretrained=True)
loss = torch.tensor([1.0], requires_grad=True)
loss.backward()
print('end')