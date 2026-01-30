import torchvision
import torch

model = torchvision.models.resnet18()
example = torch.rand(1,3,224,224)
my_torchscript_module = torch.jit.trace(model, example)
torch.jit.save(my_torchscript_module, "sciptedModule.pt")