# torch.rand(B, C, H, W, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape (B: batch size, C: channels, H: height, W: width)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # convolutional layer (sees 298x298x3 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 3)
        # convolutional layer (sees 147x147x16 tensor)
        self.conv2 = nn.Conv2d(16, 32, 3)
        # convolutional layer (sees 71x71x32 tensor)
        self.conv3 = nn.Conv2d(32, 64, 3)
        # convolutional layer (sees 33x33x64 tensor)
        self.conv4 = nn.Conv2d(64, 64, 3)
        # convolutional layer (sees 14x14x64 tensor)
        self.conv5 = nn.Conv2d(64, 64, 3)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (64 * 7 * 7 -> 500)
        self.fc1 = nn.Linear(3136, 512)
        # linear layer (512 -> 1)
        self.fc2 = nn.Linear(512, 1)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        # flatten image input
        x = x.view(-1, 64 * 7 * 7)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer
        x = self.fc2(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Assuming batch size B=1, channels C=3, height H=298, width W=298
    B, C, H, W = 1, 3, 298, 298
    return torch.rand(B, C, H, W, dtype=torch.float32)

