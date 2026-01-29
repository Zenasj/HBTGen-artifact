# torch.rand(2, 3, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape
import torch.nn as nn
import torch

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    model = MyModel()
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.zero_()
    return model

class ActiveGradsHook:
    def __init__(self, name):
        self.name = name
        # self.__name__ = None  # THIS WILL FIX YOUR CRASH

    def __call__(self, grad):
        return grad.clone()

def train_new_neurons(model):
    # Generate hooks for each layer
    for name, param in model.named_parameters():
        hook = ActiveGradsHook(name)
        param.register_hook(hook)

    # Train simply
    train(model)

def train(model):
    inputs = GetInput()
    action_target = torch.rand(2, 2)

    action_output = model(inputs)
    action_output.sum().backward()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(2, 3, dtype=torch.float32)

