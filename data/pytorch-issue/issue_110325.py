import torchvision

import torch
def generate_data(b):
    return (
        torch.randn(b, 3, 32, 32).to(torch.float32).cuda(),
        torch.randint(1000, (b,)).cuda(),
    )

from torchvision.models import resnet18
def init_model():
    return resnet18().to(torch.float32).cuda()

model = init_model()
model_opt = torch.compile(model, dynamic=False)

for b in range(16, 32):
    data = generate_data(b)
    model_opt(data[0])