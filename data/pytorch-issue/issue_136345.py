import torch.nn as nn

tensor([[[[1.0000],
          [1.0000],
          [1.0000],
          [1.0000],
          [1.0000],
          [1.0000],
          [1.0000],
          [1.0000],
          [1.0000]],

         [[1.0000],
          [1.0000],
         ...
         [1.0000],
          [1.0000]],

         [[1.0000],
          [1.5000],
          [1.5000],
          [1.5000],
          [1.5000],
          [1.5000],
          [1.5000],
         ...
         [1.5000],
          [1.5000]]]], dtype=torch.bfloat16)

import torch

def test_cpu_hardswish():

    def backward(input):
        m = torch.nn.Hardswish()
        fwd_result = m(input)
        grad = torch.ones_like(fwd_result)
        out_bwd = fwd_result.backward(grad)
        return input.grad

    shape = (1, 9, 9, 1)
    cpu_input = torch.ones(shape, dtype=torch.bfloat16)
    cpu_input = cpu_input * 3
    cpu_input_copy = cpu_input.detach().clone()
    cpu_input.requires_grad = True
    wrapped_fn = backward

    cpu_output = wrapped_fn(cpu_input)
    torch.set_printoptions(threshold=10_000)
    print(cpu_output)

test_cpu_hardswish()