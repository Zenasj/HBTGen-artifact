import torch
import torchvision.models as models

inp = torch.randn(128, 3, 224, 224)
model = models.__dict__['resnet50'](pretrained=True).eval()
model = torch.jit.script(model)
model = torch.jit.optimize_for_inference(model)
out1 = model(inp)
model.save("resent50_model.pt")
loaded_mod = torch.jit.load("resent50_model.pt")
out2 = loaded_mod(inp)