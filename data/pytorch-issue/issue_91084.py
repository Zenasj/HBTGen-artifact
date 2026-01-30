import torch

model = ssdlite320_mobilenet_v3_large(pretrained=True)
device = torch.device('cuda')

model.to(device).eval()
model= torch.compile(model)
input_data = torch.randn(3, 512, 512)
with torch.no_grad():
    out = model([input_data.to(device)])