import torch

# Create a complex tensor with real and imaginary parts
real_part = torch.tensor([1.0, 2.0, 3.0])
imaginary_part = torch.tensor([4.0, 5.0, 6.0])
complex_tensor = torch.view_as_complex(torch.stack([real_part, imaginary_part], dim=-1)).to("mps")

# Perform an arithmetic operation 
x = 2 * complex_tensor