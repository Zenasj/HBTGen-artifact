import torch

inputs = torch.randn(1, 3, 224, 224, device='cuda')
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model = model.cuda()
model.eval()
with torch.no_grad():
    res = model(inputs)

compiled_model = torch.compile(model, backend='inductor')
with torch.no_grad():
    compiled_out = compiled_model(inputs)
torch.testing.assert_close(res, compiled_out)