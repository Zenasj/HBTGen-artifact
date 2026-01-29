# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Constant tensor from the ONNX model's missing input (node 1057)
        self.constant = nn.Parameter(torch.randn(1, 3, 224, 224))

    def forward(self, x):
        # Cast to float32 then float64 as per ONNX casts
        x_float32 = x.float()
        x_float64 = x_float32.double()

        # Compute 1/x_float64 (Div operation with constant 1.0)
        a = 1.0 / x_float64

        # Multiply by 224 (Mul with constant 224.0)
        b = a * 224.0

        # Cast back to float32 (node 654)
        b_float32 = b.float()

        # Square the result (self-Mul node 655)
        c = b_float32 * b_float32

        # Unsqueeze along axis 0 for all tensors to prepare for concatenation
        x_unsqueezed = x.unsqueeze(0)          # Shape: (1, B, 3, 224, 224)
        c_unsqueezed = c.unsqueeze(0)          # Shape: (1, B, 3, 224, 224)
        const_unsqueezed = self.constant.unsqueeze(0)
        # Expand constant to match batch dimension
        const_unsqueezed = const_unsqueezed.expand(1, x.size(0), -1, -1, -1)

        # Concatenate along axis 0 (ONNX's Concat node 660)
        outputs = torch.cat([x_unsqueezed, const_unsqueezed, c_unsqueezed], dim=0)
        return outputs

def my_model_function():
    return MyModel()

def GetInput():
    # Input shape (B, 3, 224, 224) with float32 dtype (matches Cast operations)
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

