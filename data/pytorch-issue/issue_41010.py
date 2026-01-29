# torch.rand(B, 784, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MyModel, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)  # Ensure softmax is applied along the correct dimension

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.softmax(out)
        return out

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    input_size = 784  # 28 * 28
    hidden_size = 500
    num_classes = 10
    return MyModel(input_size, hidden_size, num_classes).cuda()  # Use .cuda() to move the model to GPU

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    batch_size = 64  # Example batch size
    input_size = 784  # 28 * 28
    return torch.rand(batch_size, input_size, dtype=torch.float32).cuda()  # Generate input on GPU

