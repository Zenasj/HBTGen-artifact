import torch
import torch.nn as nn
import torch.nn.functional as F


class Localizer(nn.Module):
    def __init__(self):
        super(Localizer, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, bias=False)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, bias=False)
        self.fc1 = nn.Linear(8 * 8 * 16, 32)
        self.fc2 = nn.Linear(32, 2 * 3)

        nn.init.normal_(self.fc1.weight, 0, 1e-5)
        self.fc2.bias.data.copy_(torch.tensor([1., 0., 0., 0., 1., 0.]))

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 3)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, 8 * 8 * 16)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x.view(-1, 2, 3)


class STN(nn.Module):
    def __init__(self, localizer):
        super(STN, self).__init__()

        self.localizer = localizer

    def forward(self, x):
        theta = self.localizer(x)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x


# OK
net = Localizer()
torch.jit.trace(torch.rand(16, 1, 64, 64))(net)

# OK
net = STN(Localizer())
output = net(torch.rand(16, 1, 64, 64))
print(output.shape)

# KO
torch.jit.trace(torch.rand(16, 1, 64, 64))(net)
# torch.onnx.export(net, torch.rand(16, 1, 64, 64), "mymodel", export_params=True)