import torch.nn as nn

import torch

class ComplexSliceModel(torch.nn.Module):
    def forward(self, x):
        # Convert input to a complex tensor
        x_complex = x.to(torch.complex64)
        # Apply a slice operation on the complex tensor
        return x_complex[:, :2]

model = ComplexSliceModel()
dummy_input = torch.randn(3, 4)

# Verify the model works as expected
print("Model output:", model(dummy_input))

# This call fails due to the slice op on a complex tensor.
torch.onnx.export(model, dummy_input, "complex_slice.onnx", dynamo=True)