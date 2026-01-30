import torch
import torch.nn as nn


def hook_fn(module, input, output):
    # do anything you like
    return None


class SubM(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)
        self.register_full_backward_hook(hook_fn)
    
    def forward(self, x):
        return self.fc(x)
    
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)
        self.predictions = SubM()

    def forward(self, x):
        x = self.predictions(x)
        x = self.fc1(x)
        return self.fc2(x)

model = M()

opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

data = torch.ones(2, requires_grad = True)
labels = torch.ones(2, requires_grad = True).long()

def train(m):
    m.train()
    m = torch.compile(m, fullgraph=True)
    train_one_epoch(m)


def train_one_epoch(m):
    opt.zero_grad(True)
    output = m(data)
    loss = (output - labels).sum()
    loss.backward()
    opt.step()

train(model)