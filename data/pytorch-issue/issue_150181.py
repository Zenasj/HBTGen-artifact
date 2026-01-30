import torch
input_data = torch.randn(2, 3)
output = torch.full(size=(2, 3), fill_value=100, dtype=input_data.dtype, requires_grad=True)
torch.add(input_data, output, out=output)
# torch.sub(input_data, output, out=output)
# torch.mul(input_data, output, out=output)
# torch.div(input_data, output, out=output)