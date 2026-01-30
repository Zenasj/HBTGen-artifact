import torch
import torchvision
model = torchvision.models.alexnet(pretrained=False)
model.eval()
torch.jit.script(model).save('alexnet.pt')