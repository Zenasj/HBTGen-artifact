# torch.rand(3, 224, 224, dtype=torch.float32)  # Inferred input shape from original np.random.rand(3,224,224)
import torch
import functorch.experimental.control_flow as fc

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cond = fc.cond  # Functorch cond for controlled control flow

    def forward(self, x):
        min_val = x.min()
        max_val = x.max()
        condition = (min_val >= 0) & (max_val <= 1)
        
        def true_path(x):
            scaled = x * 255  # Apply scaling if condition met
            return scaled - scaled.mean()  # Subtract mean of scaled tensor
        
        def false_path(x):
            return x - x.mean()  # Subtract mean without scaling
        
        # Use functorch.cond to capture control flow
        return self.cond(condition, true_path, false_path, [x])[0]

def my_model_function():
    return MyModel()  # Return the fused model with controlled control flow

def GetInput():
    return torch.rand(3, 224, 224, dtype=torch.float32)  # Match input shape from original code

