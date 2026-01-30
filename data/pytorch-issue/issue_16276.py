import torch.nn as nn

# build a model with
linear = nn.Linear(64, 10)

x = self.previous_layer(some_input)
x.register_hook(my_hook_for_input)
output = self.my_linear(x)

model = nn.Sequential(
        nn.Linear(28*28, 300),
        nn.ReLU(),
        nn.Linear(300, 10)  # Wx+b - I want grad_input w.r.t x here
    )

def hook_on_forward(mod, input):
    input.register_hook(my_hook_for_input)

model[-1].register_forward_pre_hook(hook_on_forward)