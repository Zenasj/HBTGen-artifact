import torch
import torchvision.models as models

model = models.__dict__['googlenet'](pretrained=True)
input_tensor = torch.randn(1, 3, 224, 224)
model = model.eval()
input_tensor = input_tensor.to(memory_format=torch.channels_last)
model = model.to(memory_format=torch.channels_last)
with torch.cpu.amp.autocast(cache_enabled=False), torch.no_grad():
    model = torch.jit.trace(model, input_tensor, strict=False)
model = torch.jit.freeze(model)

with torch.no_grad():
    for i in range(10):
        output = model(input_tensor)
        print(output)