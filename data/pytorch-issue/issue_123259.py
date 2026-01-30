import torch.nn as nn

import torch 

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(1, 1)

    def forward(self, x):
        x = self.linear1(x)
        return x
    
# Set device and dtype
device = torch.device("mps:0")
dtype = torch.float32

# set models
model = Model()
criterion = torch.nn.L1Loss()
model.to(device)

# set non-trainable params
data = torch.tensor([[0.1]], dtype= dtype, device = device)
target = torch.tensor([[0.1]], dtype= dtype, device = device)

state_dict = model.state_dict()

# take trainable params
weight = state_dict["linear1.weight"]
bias = state_dict["linear1.bias"]

# calculate output of the model manually
calculated_linear_output = weight * data + bias

# find output of the torch model
output = model(data)

# check if two outputs are the same (there is no problem in here both in MPS and CPU)
assert output == calculated_linear_output

# calculate loss and gradients
loss = criterion(output, target)
loss.backward()

# calculate expected weight and biases manually 
expected_gradient_b = torch.sign(output - target)
expected_gradient_w = torch.sign(output - target) * data
grad_list = [expected_gradient_w, expected_gradient_b]

# check if all calculated gradients are the same (fails in MPS and not fails in CPU)
for param, my_param in zip(model.parameters(), grad_list):
    assert param.grad == my_param

import torch 

device = torch.device("mps:0")
dtype = torch.float32

def linear_1d(a,x,b):
    return a @ x + b
criterion = torch.nn.L1Loss()

data = torch.tensor([[2.0]], device= device, dtype = dtype)
weight = torch.tensor([[1.0]], device= device, dtype = dtype, requires_grad = True)
bias = torch.tensor([[0.7]], device= device, dtype = dtype, requires_grad = True)
target = torch.tensor([[0.3]], device= device, dtype = dtype)

output = linear_1d(weight, data, bias)
loss = criterion(output, target)
loss.backward()

expected_weight_gradients =  torch.sign(output - target) * data
expected_bias_gradients =  torch.sign(output - target)

assert weight.grad == expected_weight_gradients
assert bias.grad == expected_bias_gradients