import torchvision

import torch
from torchvision.models import resnet18

model = resnet18().cuda().half().eval()
test_data = torch.randn(1, 3, 224, 224, requires_grad=False).half().cuda()
o1 = model(test_data)
torch.save(model.state_dict(), 'model.pkl')

model_reload = resnet18().cuda().eval()
model_reload.load_state_dict(torch.load('model.pkl'))
o2 = model_reload(test_data.float())

print(o1[0, :10])
print(o2[0, :10])