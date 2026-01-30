import torch

image1 = torch.randn(1, 3, 224, 192)
image2 = torch.randn(1, 3, 112, 112)
model = FullModel(num_ch=28)

weight = torch.load('model.pth', map_location='cpu')
model.generator.load_state_dict(weight['g'])
#model.generator.eval()
model.encoder.load_state_dict(weight['e'])
#model.encoder.eval()
model.eval()
with torch.no_grad():
    torch.onnx.export(model, (image1, image2), 'test.onnx', opset_version=11)