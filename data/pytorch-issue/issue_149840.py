import torch
import torch._dynamo.testing

device = "cpu"

cnt = torch._dynamo.testing.CompileCounter()

def m(input):
  for i in range(8):
    input = input * 3
  return input

m = torch.compile(m, backend=cnt)

input = torch.zeros(1, 128, dtype=torch.bfloat16).to(device)
output = m(input)

print(cnt.frame_count)

# Disable dynamo
disable = os.environ.get("TORCH_COMPILE_DISABLE", False)