import torch
import torch.nn as nn

# torch.rand(B, 3, 28, 28, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True)
        self.fc = nn.Linear(16 * 28 * 28, 10, bias=True)
        self.conv.register_backward_hook(self.conv_backward_hook)
        self.fc.register_backward_hook(self.fc_backward_hook)
        self.conv_grad_input = None
        self.fc_grad_input = None
        self.inconsistent = False  # Flag to track discrepancies

    def conv_backward_hook(self, module, grad_input, grad_output):
        self.conv_grad_input = grad_input  # Capture gradients for comparison
        return grad_input  # Return unchanged gradients

    def fc_backward_hook(self, module, grad_input, grad_output):
        self.fc_grad_input = grad_input  # Capture gradients for comparison
        return grad_input  # Return unchanged gradients

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def check_inconsistencies(self):
        # Check bias gradient positions and weight gradient shapes
        if not (self.conv_grad_input and self.fc_grad_input):
            return False  # No gradients captured yet

        # Check bias gradient positions (conv: last element, fc: first element)
        if len(self.conv_grad_input) < 3 or len(self.fc_grad_input) < 3:
            self.inconsistent = True
            return self.inconsistent
        if self.conv_grad_input[2] is None or self.fc_grad_input[0] is None:
            self.inconsistent = True
            return self.inconsistent

        # Check weight gradient shapes (conv: matches weights, fc: transpose of weights)
        conv_weight_shape = self.conv_grad_input[1].shape
        if conv_weight_shape != self.conv.weight.shape:
            self.inconsistent = True
        fc_weight_shape = self.fc_grad_input[2].shape
        expected_fc_shape = (self.fc.out_features, self.fc.in_features)
        if fc_weight_shape != expected_fc_shape:
            self.inconsistent = True

        # Check bias gradient shapes (conv: (out_channels,), fc: (batch_size, out_features))
        if self.conv_grad_input[2].shape != (self.conv.out_channels,):
            self.inconsistent = True

        return self.inconsistent

def my_model_function():
    return MyModel()

def GetInput():
    B = 2  # Arbitrary batch size
    return torch.rand(B, 3, 28, 28, dtype=torch.float32)

