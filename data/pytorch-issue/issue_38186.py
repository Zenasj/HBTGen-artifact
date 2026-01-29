# torch.rand(B, C, H, W, dtype=torch.float32)  # Shape: (1, 1, 512, 229)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc_onset = nn.Linear(64 * 229 * 512, 88)  # Matches output dimensions from error report
        self.fc_frame = nn.Linear(64 * 229 * 512, 88)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # Flatten for FC layers
        onset = self.fc_onset(x)
        frame = self.fc_frame(x)
        # Return dummy intermediate tensors to match 4-element tuple requirement
        return (onset, x, x, frame)  # Matches iOS extraction of indices 0 and 3

def my_model_function():
    model = MyModel()
    # Initialize weights to match training setup (assumed He initialization)
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    return model

def GetInput():
    # Input dimensions based on iOS code: {1, 1, frameCount, 229}
    # Using frameCount=512 as common audio frame size
    return torch.rand(1, 1, 512, 229, dtype=torch.float32)

