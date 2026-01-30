import torch

_, code = run_and_get_cpp_code(
      torch._dynamo.optimize("inductor")(fn),
      x,
  )