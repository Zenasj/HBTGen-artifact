import torch  # '1.12.0.dev20220426+cu113' and '1.12.0a0+gitb17b2b1'
import torchvision  # 0.13.0a0+01b0a00

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
model.eval()

smodel = torch.jit.script(model)
smodel.eval()
smodel([torch.rand(3, 224, 224), ])