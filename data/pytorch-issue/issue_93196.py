model = models.resnet18()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
compiled_model = torch.compile(model)
start = time.time()
x = torch.randn(16, 3, 224, 224)
optimizer.zero_grad()
out = compiled_model(x)
out.sum().backward()
optimizer.step()
end = time.time()
print(end-start)

import torch
import torch._dynamo as dynama
import torchvision.models as models
import time