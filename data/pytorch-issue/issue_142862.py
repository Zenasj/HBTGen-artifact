import torch
import torch.nn as nn

class mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 10)
        self.layer4 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

recompute_list = [torch.ops.aten.addmm.default]
def recompute_policy(ctx, op, *args, **kwargs):
    if op in recompute_list:
        return CheckpointPolicy.MUST_RECOMPUTE
    else:
        return CheckpointPolicy.PREFER_SAVE

def context_fn():
    return create_selective_checkpoint_contexts(recompute_policy)

model = mlp().cuda()
input = torch.randn(1, 10).cuda()

for i in range(1):
    out = torch.utils.checkpoint.checkpoint(model, input, use_reentrant=False, context_fn=context_fn)
    with torch._dynamo.compiled_autograd._enable(torch.compile(backend=my_compiler, fullgraph=True)):
        out.sum().backward()