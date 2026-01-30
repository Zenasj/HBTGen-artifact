import torch
import functorch

# Reduced from test_torch.py: test_exponential
def poc7():
  device = 'cpu'

  test = (-0, float('inf'))
  t = torch.empty((1,), device=device, dtype=torch.bfloat16).exponential_(test[0])
  print(t.item() == test[1])


print("CPU")
poc7()
print()

print("CPU Functionalize")
functorch.functionalize(poc7)()