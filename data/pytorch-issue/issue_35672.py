import torch
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10,2)
        self.fc2 = nn.Linear(10,2)

    def forward(self, x):
        return self.fc1(x), self.fc2(x)

def model_forward(model, x):
    o1, o2 = model(x) # two variables are returned
    return o1

x = torch.rand(2, 10)
model = TestModel()
model = torch.nn.DataParallel(model).cuda() # to multi-gpu

x = x.cuda()
o1 = model_forward(model, x) # a function to warp the forward process
o1.sum().backward()

for p in model.parameters():
    print(p.grad)

tensor([[0.4842, 1.5574, 1.1879, 0.5454, 0.9086, 1.2837, 1.3273, 0.4979, 0.3307,
         0.4577],
        [0.4842, 1.5574, 1.1879, 0.5454, 0.9086, 1.2837, 1.3273, 0.4979, 0.3307,
         0.4577]], device='cuda:0')
tensor([2., 2.], device='cuda:0')
tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], device='cuda:0')
tensor([0., 0.], device='cuda:0')