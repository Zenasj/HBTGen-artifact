import torch

@torch.no_grad()
def profile_model(model):
    y = torch.randn(1,3,224,224).cuda()
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        model(y)
    return prof

x = torch.randn(1,3,224,224).cuda()

model = resnet18(pretrained=True).cuda()
profile_model(model)
model.eval()
print(model(x).norm())

model = resnet18(pretrained=True).cuda()
model.eval()
print(model(x).norm())