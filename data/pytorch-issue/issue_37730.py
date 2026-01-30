import torch.nn as nn
import torchvision

import torch
from torchvision.models import resnet18

amp_enabled = True
checkpoint_enabled = True
device = torch.device('cuda')

class ResNetWithCheckpoints(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = resnet18()

    def forward(self, x):
        base = self.base
        x = base.conv1(x)
        x = base.bn1(x)
        x = base.relu(x)
        x = base.maxpool(x)

        if checkpoint_enabled:
            x = torch.utils.checkpoint.checkpoint(base.layer1, x)
            x = torch.utils.checkpoint.checkpoint(base.layer2, x)
            x = torch.utils.checkpoint.checkpoint(base.layer3, x)
            x = torch.utils.checkpoint.checkpoint(base.layer4, x)
        else:
            x = base.layer1(x)
            x = base.layer2(x)
            x = base.layer3(x)
            x = base.layer4(x)

        x = base.avgpool(x)
        x = torch.flatten(x, 1)
        x = base.fc(x)

        return x

model = ResNetWithCheckpoints()
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

xs = torch.randn(5, 3, 128, 128).to(device)
ys = torch.tensor(range(len(xs))).to(device)
for _ in range(20):
    optimizer.zero_grad()
    with torch.cuda.amp.autocast(enabled=amp_enabled):
        ys_pred = model(xs)
        loss = loss_fn(ys_pred, ys)
    print(float(loss))
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()