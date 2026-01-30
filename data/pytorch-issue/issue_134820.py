import torch 

class toy_fn(torch.autograd.Function):
  @staticmethod
  def forward(ctx,inputs):
    torch.exp(inputs, out=inputs)
    return inputs

  @staticmethod
  def backward(ctx, grad_output):
    return None, None

@torch.compile(fullgraph=True)
def compile_repro(inputs):
    l = toy_fn.apply(inputs)
    return l

input = torch.randn(128,1, requires_grad=True)

compile_repro(input)