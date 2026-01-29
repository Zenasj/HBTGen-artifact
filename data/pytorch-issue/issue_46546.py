# torch.rand(B, 5, dtype=torch.complex64)  # Inferred input shape (batch_size, 5 features, complex64)
import torch
import torch.nn as nn
import math

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ComplexLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.complex64))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.complex64))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return torch.nn.functional.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = ComplexLinear(5, 5)  # Matches the example's input/output dimensions

    def forward(self, x):
        return self.layer(x)

def my_model_function():
    return MyModel()  # Returns initialized MyModel instance

def GetInput():
    return torch.rand(1, 5, dtype=torch.complex64)  # Matches model's input requirements

