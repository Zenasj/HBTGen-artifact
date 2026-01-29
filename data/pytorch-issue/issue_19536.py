# torch.rand(B, C, H, W, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn
import math

class MyModel(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.5):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(out_features, out_features)
        
        # Initialize the weights of the first linear layer using the new dropout initialization
        self._initialize_weights(self.fc1, dropout_rate)

    def _initialize_weights(self, module, p):
        if isinstance(module, nn.Linear):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            gain = math.sqrt(2 * (1 - p))
            std = gain / math.sqrt(fan_in)
            with torch.no_grad():
                module.weight.normal_(0, std)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel(in_features=100, out_features=50, dropout_rate=0.5)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B, C, H, W = 1, 1, 10, 10  # Example batch size, channels, height, and width
    return torch.rand(B, C, H, W, dtype=torch.float32).view(B, -1)  # Flatten the input to match the linear layer

