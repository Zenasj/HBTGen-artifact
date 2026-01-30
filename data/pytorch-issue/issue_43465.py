import torch

py
with torch.autograd.profiler.profile(profile_memory=True) as prof1:
    a = torch.empty(10000000, device="cuda")
    del a
print(prof1.table())

py
class DummyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        a = torch.empty(10000000, device="cuda")
        del a
        return args

inp = torch.empty(1, device="cuda")
with torch.autograd.profiler.profile(profile_memory=True) as prof2:
    DummyFunction.apply(inp)
print(prof2.table())