import torch
import torch.nn as nn


# Define a simple multi - layer perceptron
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Apply checkpointing to the first fully - connected layer
        def custom_forward(*inputs):
            layer, input_tensor = inputs
            return layer(input_tensor)
        # error of module 'torch.utils' has no attribute 'checkpoint' here
        out = torch.utils.checkpoint.checkpoint(custom_forward, self.fc1, x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Set the input parameters
input_size = 5
hidden_size = 10
output_size = 5
batch_size = 4

# Create an instance of the MLP
model = SimpleMLP(input_size, hidden_size, output_size)

# Generate some random input data
x = torch.randn(batch_size, input_size, requires_grad=True)

# Forward pass
output = model(x)

# Compute the loss (using a simple sum as an example)
loss = output.sum()

# Backward pass
loss.backward()

# Print the gradients of the input tensor
print("Gradients of the input tensor:", x.grad)