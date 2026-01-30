import torch.nn as nn

import torch

class Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.randn((16,))
        self.variance_epsilon = 1e-5

    @torch.compile
    def forward(self, hidden_states, residuals=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(-1, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states -
                            mean) * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight.to(torch.float32) * hidden_states
        return hidden_states.to(input_dtype), residuals

layers = [Layer() for i in range(100)]
hidden_states = torch.randn((32, 16, 16))

for iteration in range(2):
    # simulate a model forward call
    for layer in layers:
        hidden_states, _ = layer(hidden_states)