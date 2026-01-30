# mini case
import torch
import torch.nn as nn

b = 2
value = torch.ones(b, 128, 512, 256, requires_grad=True).cuda()
value.grad = torch.ones_like(value)

model = nn.Conv2d(128, 128, 3, padding=1, groups=128).cuda().train()

res = model(value)
res.backward(torch.ones_like(res))
value_grad = model.weight.grad.detach().clone()

model.weight.grad = None

model_compile = torch.compile(model)

res_ = model_compile(value)
res_.backward(torch.ones_like(res_))
value_grad_ = model.weight.grad.detach().clone()
print(f"Total diff: {torch.mean(res - res_)}, Max diff: {torch.max(res - res_)}")
print(f"Total grad diff: {torch.sum(value_grad - value_grad_)}")
assert torch.allclose(res, res_)
assert value_grad.allclose(value_grad_)