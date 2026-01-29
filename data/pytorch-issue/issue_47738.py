# torch.rand(1, 3, 64, 64, dtype=torch.float32) for first input and torch.tensor(4.0, dtype=torch.float32) for second
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        weight_tensor = torch.empty(3,21,16,64)
        init_weight = torch.nn.init.xavier_uniform_(weight_tensor)
        self.w_matrix = nn.Parameter(init_weight)

    def forward(self, x):
        num_out, num_in, _, _ = self.w_matrix.shape
        b = torch.zeros(x.shape[0], num_out, num_in)
        return b

class FixedNet(nn.Module):
    def __init__(self):
        super().__init__()
        weight_tensor = torch.empty(3,21,16,64)
        init_weight = torch.nn.init.xavier_uniform_(weight_tensor)
        self.w_matrix = nn.Parameter(init_weight)

    def forward(self, x):
        num_out, num_in, _, _ = self.w_matrix.shape
        num_out = int(num_out)
        num_in = int(num_in)
        b = torch.zeros(x.shape[0], num_out, num_in)
        return b

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros([5]))

    def forward(self, x):
        return x + self.p.size()[0]

class FixedMyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros([5]))

    def forward(self, x):
        return x + int(self.p.size(0))

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_a = Net()          # Original model from first example
        self.fixed_net_a = FixedNet()  # Fixed version
        self.net_b = MyModule()     # Original model from second example
        self.fixed_net_b = FixedMyModule()  # Fixed version

    def forward(self, inputs):
        x1, x2 = inputs
        # Run original and fixed versions for first example
        out_a = self.net_a(x1)
        fixed_out_a = self.fixed_net_a(x1)
        # Run original and fixed versions for second example
        out_b = self.net_b(x2)
        fixed_out_b = self.fixed_net_b(x2)
        # Compare outputs using torch.allclose with tolerance
        compare_a = torch.allclose(out_a, fixed_out_a, atol=1e-5)
        compare_b = torch.allclose(out_b, fixed_out_b, atol=1e-5)
        # Return combined result as tensor
        return torch.tensor(compare_a and compare_b, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    x1 = torch.rand(1, 3, 64, 64, dtype=torch.float32)
    x2 = torch.tensor(4.0, dtype=torch.float32)
    return (x1, x2)

