import torch
import torchvision

model = torchvision.models.alexnet()
model.classifier = torch.jit.script(model.classifier)

w = SummaryWriter()
w.add_graph(model, torch.rand((2, 3, 224, 224)))
w.close()