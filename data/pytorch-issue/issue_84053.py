import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10))
        self.register_buffer('input_mean', torch.tensor(0.))

    def forward(self, x):
        self.input_mean = 0.9 * self.input_mean + 0.1 * x.mean()
        return self.mlp(x.flatten(1) / self.input_mean)

model = ToyModel()
ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
        0.05 * averaged_model_parameter + 0.95 * model_parameter
ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)

optimzier = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimzier, milestones=[2])
train_dataset = datasets.MNIST(root="MNIST", download=True, train=True, transform=transforms.ToTensor())
train_dataloader = DataLoader(dataset=train_dataset, batch_size=100, num_workers=2)

for epoch in range(3):
    for input, target in train_dataloader:
        x = model(input)
        loss = F.cross_entropy(x, target)
        optimzier.zero_grad()
        loss.backward()
        optimzier.step()
        ema_model.update_parameters(model)

print(ema_model.module.input_mean)
print(model.input_mean)