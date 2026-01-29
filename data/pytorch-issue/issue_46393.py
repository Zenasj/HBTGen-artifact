# torch.rand(1, 28, 28, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
        # Define convolutional layers for different kernel sizes
        self.conv11 = nn.Conv2d(1, 16, 3, 1)  # Input = 1x28x28  Output = 16x26x26
        self.conv12 = nn.Conv2d(1, 16, 5, 1)  # Input = 1x28x28  Output = 16x24x24
        self.conv13 = nn.Conv2d(1, 16, 7, 1)  # Input = 1x28x28  Output = 16x22x22

        self.conv21 = nn.Conv2d(16, 32, 3, 1)  # Input = 16x26x26 Output = 32x24x24
        self.conv22 = nn.Conv2d(16, 32, 5, 1)  # Input = 16x24x24 Output = 32x20x20
        self.conv23 = nn.Conv2d(16, 32, 7, 1)  # Input = 16x22x22 Output = 32x16x16

        self.conv31 = nn.Conv2d(32, 64, 3, 1)  # Input = 32x24x24 Output = 64x22x22
        self.conv32 = nn.Conv2d(32, 64, 5, 1)  # Input = 32x20x20 Output = 64x16x16
        self.conv33 = nn.Conv2d(32, 64, 7, 1)  # Input = 32x16x16 Output = 64x10x10

        # Define max pooling layer
        self.maxpool = nn.MaxPool2d(2)  # Output = 64x11x11, 64x8x8, 64x5x5

        # Define dropout layers
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # Define fully connected layers
        self.fc11 = nn.Linear(64 * 11 * 11, 128)
        self.fc12 = nn.Linear(64 * 8 * 8, 128)
        self.fc13 = nn.Linear(64 * 5 * 5, 128)

        self.fc21 = nn.Linear(128, 10)
        self.fc22 = nn.Linear(128, 10)
        self.fc23 = nn.Linear(128, 10)

        self.fc33 = nn.Linear(30, 10)

    def forward(self, x1):
        # Branch 1
        x = F.relu(self.conv11(x1))
        x = F.relu(self.conv21(x))
        x = F.relu(self.maxpool(self.conv31(x)))
        x = x.view(-1, 64 * 11 * 11)
        x = self.dropout1(x)
        x = F.relu(self.fc11(x))
        x = self.dropout2(x)
        x = self.fc21(x)

        # Branch 2
        y = F.relu(self.conv12(x1))
        y = F.relu(self.conv22(y))
        y = F.relu(self.maxpool(self.conv32(y)))
        y = y.view(-1, 64 * 8 * 8)
        y = self.dropout1(y)
        y = F.relu(self.fc12(y))
        y = self.dropout2(y)
        y = self.fc22(y)

        # Branch 3
        z = F.relu(self.conv13(x1))
        z = F.relu(self.conv23(z))
        z = F.relu(self.maxpool(self.conv33(z)))
        z = z.view(-1, 64 * 5 * 5)
        z = self.dropout1(z)
        z = F.relu(self.fc13(z))
        z = self.dropout2(z)
        z = self.fc23(z)

        # Concatenate and final layer
        out = self.fc33(torch.cat((x, y, z), dim=1))
        output = F.log_softmax(out, dim=1)
        return output

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 1, 28, 28, dtype=torch.float32)

