# torch.rand(B, 3, 120, 120, 120, dtype=torch.float)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv_layer1 = self._conv_layer_set(3, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        self.conv_layer3 = self._conv_layer_set(64, 128)
        self.conv_layer4 = self._conv_layer_set(128, 256)
        self.conv_layer5 = self._conv_layer_set(256, 512)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 2)  # num_classes=2 inferred from context
        self.relu = nn.ReLU()
        self.batch = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=0.6, inplace=True)
    
    def _conv_layer_set(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )
    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.conv_layer5(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, 120, 120, 120, dtype=torch.float)

