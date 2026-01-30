import torch

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
device = torch.device('cuda')

model.to(device).eval()
model = torch.compile(model, backend='onnxrt')
input_data = torch.randn((6, 3, 224, 224))
input_data = input_data.to("cuda")
output_data = model(input_data)

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
device = torch.device('cuda')

model.to(device).eval()
model = torch.compile(model, backend='torch2trt')
input_data = torch.randn((6, 3, 224, 224))
input_data = input_data.to("cuda")
output_data = model(input_data)