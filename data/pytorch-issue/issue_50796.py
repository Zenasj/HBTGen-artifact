import torch
import torchvision
import time

input = torch.randn(1, 3, 224, 224).cuda()
model = torchvision.models.resnet18().cuda()
model.eval()
_ = model(input)
traced_model = torch.jit.trace(model, input, strict=False)
torch.jit.save(traced_model, './jit_trace_model.pt')

model = torch.jit.load('./jit_trace_model.pt').cuda()
input = torch.randn(1, 3, 224, 224).cuda()

model.eval()
for _ in range(4):
    time0 = time.time()
    _ = model(input)
    time1 = time.time()
    print(time1 - time0)