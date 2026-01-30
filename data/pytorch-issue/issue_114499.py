import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLayer(nn.Module):
    def __init__(self, size):
        super(CustomLayer, self).__init__()
        # Initialize a constant tensor for element-wise multiplication
        self.weights = nn.Parameter(torch.randn(size), requires_grad=True)

    def forward(self, x):
        # Element-wise multiplication
        return x * self.weights

class SampleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SampleNN, self).__init__()
        self.custom_layer = CustomLayer(input_size)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.custom_layer(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

input_size = 10
hidden_size = 20
output_size = 5

sample_nn = SampleNN(input_size, hidden_size, output_size)

def my_backend(gm, inputs):
    print(gm)
    # breakpoint()
    return gm

nn = torch._dynamo.optimize(my_backend, remote=True)(sample_nn)
nn(torch.randn([1, input_size]))
nn(torch.randn([1, input_size]))
nn(torch.randn([1, input_size]))
exit(0)

class CustomLayer(nn.Module):
    def __init__(self, size):
        super(CustomLayer, self).__init__()
        # Initialize a constant tensor for element-wise multiplication
        self.weights = nn.Parameter(torch.randn(size), requires_grad=True)

    def forward(self, x):
        # Element-wise multiplication
        return x * self.weights

class SampleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SampleNN, self).__init__()
        self.custom_layer = CustomLayer(input_size)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.custom_layer(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x) * 2
        print("woowoo")
        x = x * x
        return x

input_size = 10
hidden_size = 20
output_size = 5

sample_nn = SampleNN(input_size, hidden_size, output_size)

def my_backend(gm, inputs):
    print(gm)
    return gm

nn = torch._dynamo.optimize(my_backend, serialize=True)(sample_nn)
nn(torch.randn([1, input_size]))
nn(torch.randn([1, input_size]))
nn(torch.randn([1, input_size]))