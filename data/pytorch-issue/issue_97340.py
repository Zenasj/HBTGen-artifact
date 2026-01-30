import torch
import torchvision

device = "cuda:0"

x = torch.rand(1, 3, 800, 800, device=device)

model = torchvision.models.detection.maskrcnn_resnet50_fpn()
model = model.to(device)
model = torch.compile(model)
model.eval()

# This works
# _ = model(x)

# This does not
with torch.no_grad():
    _ = model(x)