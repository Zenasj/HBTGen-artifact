import torch
from torch.nn import LayerNorm, Linear
from torch.jit import ScriptModule, script_method

class Test(ScriptModule):
    def __init__(self, dim):
        super().__init__()
        self.layer_norm = LayerNorm(dim)
        self.projection = Linear(dim, dim)
    
    @script_method
    def forward(self, inputs):
        return self.layer_norm(inputs + self.projection(inputs))


if __name__ == "__main__":
    m = Test(512)
    input_tensor = torch.randn((10, 11, 512))
    output_tensor = m(input_tensor)
    output_tensor.sum().backward()