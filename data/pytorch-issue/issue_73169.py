import torch
results = dict()
input_tensor = torch.rand([3, 3], dtype=torch.complex128)
out_tensor = torch.rand([3, 3], dtype=torch.float64)

print(torch.abs(input_tensor.clone(), out=out_tensor.clone()))
# tensor([[0.2063, 1.2438, 1.3721],
#      [0.6465, 1.1026, 0.7529],
#      [1.2668, 1.0570, 0.3174]], dtype=torch.float64)
torch.arccos(input_tensor.clone(), out=out_tensor.clone())
# RuntimeError: result type ComplexDouble can't be cast to the desired output type Double