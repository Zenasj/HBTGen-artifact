import torch
import torch.nn as nn

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('epsilon_input', torch.zeros(4, 4))

    def forward(self, foo):
        torch.randn(self.epsilon_input.size(), out=self.epsilon_input)
        return foo + self.epsilon_input

x = torch.ones([4, 4])
model = MyModule()
opt_model = torch._dynamo.optimize("eager")(model)
print(opt_model(x))