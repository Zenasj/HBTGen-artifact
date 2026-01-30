import torch
import torchvision.models as models

model = models.resnet18().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
compiled_model = torch.compile(model)

x = torch.randn(16, 3, 224, 224).cuda()
optimizer.zero_grad()
out = compiled_model(x)
out.sum().backward()
optimizer.step()

compiled_model = torch.compile(model, fullgraph=True, backend='nvprims_aten')

import torch._dynamo as dynamo
print(dynamo.list_backends())