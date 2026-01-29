# torch.rand(B, 1, H, W, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.dropout2 = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(self.dropout1(x), 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def log_weights(self, step, writer):
        if isinstance(self, nn.DataParallel):
            model = self.module
        else:
            model = self
        writer.add_histogram('weights/conv1/weight', model.conv1.weight.data, step)
        writer.add_histogram('weights/conv1/bias', model.conv1.bias.data, step)
        writer.add_histogram('weights/conv2/weight', model.conv2.weight.data, step)
        writer.add_histogram('weights/conv2/bias', model.conv2.bias.data, step)
        writer.add_histogram('weights/fc1/weight', model.fc1.weight.data, step)
        writer.add_histogram('weights/fc1/bias', model.fc1.bias.data, step)
        writer.add_histogram('weights/fc2/weight', model.fc2.weight.data, step)
        writer.add_histogram('weights/fc2/bias', model.fc2.bias.data, step)

def my_model_function():
    return MyModel()

def GetInput():
    # Assuming batch size (B) is 1, and input image size (H, W) is 28x28
    B, C, H, W = 1, 1, 28, 28
    return torch.rand(B, C, H, W, dtype=torch.float32)

