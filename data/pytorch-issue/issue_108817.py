# torch.rand(3, 4, dtype=torch.float32)
import torch
import functorch.experimental.control_flow as control_flow

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('buffer', torch.ones(6, 4))  # From second example's buffer

    def forward(self, x):
        # First scenario: cond with compatible return types
        def true1(x): return x.sin()
        def false1(x): return x  # Adjusted to return tensor instead of tuple
        pred1 = torch.tensor(True)  # Example predicate
        res1 = control_flow.cond(pred1, true1, false1, [x])
        
        # Second scenario: cond without in-place buffer modification
        def true2(x):
            new_buffer = self.buffer + 1  # Non-inplace modification
            return new_buffer.sum() + x.sum()
        def false2(x): return x.sum()
        pred2 = torch.tensor(x.shape[0] > 4)  # Dynamic predicate from input
        res2 = control_flow.cond(pred2, true2, false2, [x])
        
        return res1, res2

def my_model_function():
    return MyModel()

def GetInput():
    return torch.ones(3, 4)  # Matches input shape (3,4) from examples

