import torch

def foo():
  a = torch.randint(-1, 0, (1, 1, 1), device="cpu", dtype=torch.int64)
  res_cpu = torch.bitwise_right_shift(a, 31)
  print("cpu is" , res_cpu)

compiled_foo = torch.compile(foo)
compiled_foo()