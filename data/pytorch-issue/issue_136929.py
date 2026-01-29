# torch.rand(B, C, H, W, dtype=...)  # Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        return self.linear2(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    model = MyModel()
    
    # Set custom attributes on the parameters
    model.linear1.weight.custom_attr = 'custom_value_linear1_weight'
    model.linear1.bias.custom_attr = 'custom_value_linear1_bias'
    model.linear2.weight.custom_attr = 'custom_value_linear2_weight'
    model.linear2.bias.custom_attr = 'custom_value_linear2_bias'
    
    return model

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Assuming the input shape is (batch_size, 10) for the linear layer
    batch_size = 4  # Example batch size
    return torch.rand(batch_size, 10, dtype=torch.float32)

