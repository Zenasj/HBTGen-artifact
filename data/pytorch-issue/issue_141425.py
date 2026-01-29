# torch.rand(B, C, H, W, dtype=torch.float32) ← Input shape is (2, 3, 4) as per the example
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, eps=1e-5, dtype=torch.float32):
        super().__init__()
        self.ln = nn.LayerNorm(4, eps=eps, dtype=dtype)
        self.intermediates = None  # Stores (mean, shift, var, rstd, x_hat)
        self.x = None              # Stores input tensor
        self.out = None            # Stores output tensor

    def forward(self, x):
        self.x = x
        self.out = self.ln(x)
        
        # Compute intermediates for manual gradient calculation
        mean = x.mean(dim=-1, keepdim=True)
        shift = x - mean
        var = (shift ** 2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + self.ln.eps)
        x_hat = shift * rstd
        self.intermediates = (mean, shift, var, rstd, x_hat)
        
        return self.out

    def compare_gradients(self):
        """Compares PyTorch autograd gradients with manual gradients using 1e-5 atol"""
        auto_weight = self.ln.weight.grad
        auto_bias = self.ln.bias.grad
        auto_input = self.x.grad

        # Extract intermediates
        mean, shift, var, rstd, x_hat = self.intermediates

        # Compute manual gradients
        with torch.no_grad():
            nabla_y = self.out  # Assumes loss gradient is self.out (0.5*out²)
            hand_weight = (nabla_y * x_hat).sum((0,1))
            hand_bias = nabla_y.sum((0,1))
            
            nabla_x_hat = nabla_y * self.ln.weight
            hand_input = rstd * (
                nabla_x_hat 
                - nabla_x_hat.mean(-1, True) 
                - x_hat * (nabla_x_hat * x_hat).mean(-1, True)
            )
            
        # Convert to float64 for accurate comparison
        auto_weight = auto_weight.to(torch.float64)
        auto_bias = auto_bias.to(torch.float64)
        auto_input = auto_input.to(torch.float64)
        hand_weight = hand_weight.to(torch.float64)
        hand_bias = hand_bias.to(torch.float64)
        hand_input = hand_input.to(torch.float64)

        # Check using float32 precision tolerance
        weight_ok = torch.allclose(auto_weight, hand_weight, atol=1e-5)
        bias_ok = torch.allclose(auto_bias, hand_bias, atol=1e-5)
        input_ok = torch.allclose(auto_input, hand_input, atol=1e-5)
        
        return weight_ok and bias_ok and input_ok

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn((2, 3, 4), dtype=torch.float32, requires_grad=True, device="cuda")

