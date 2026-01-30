import torch

def forward(x, y):          
  return torch.abs(input=x, out=torch.t(input=y))

x = torch.rand([9, 10], dtype=torch.float32)
y = torch.rand([10, 9], dtype=torch.float32)

# Run in eager mode
eager = forward(x, y)

# Run in compiled mode
compiled = torch.compile(forward, mode='default')(x, y)

# Check if both results are close
result_close = torch.allclose(eager.to('cpu'), compiled.to('cpu'), rtol=1e-3, atol=1e-3, equal_nan=True)
print(f"Are the results close? {result_close}")