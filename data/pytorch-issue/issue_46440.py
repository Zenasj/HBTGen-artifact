import torch

ir = """
graph(%0 : Float(1, 256, strides=[256, 1], requires_grad=0, device=cuda:0),
      %1 : Float(1, 512, strides=[512, 1], requires_grad=0, device=cuda:0),
      %2 : Float(1, 140, strides=[140, 1], requires_grad=0, device=cuda:0),
      %3 : Float(1, 140, strides=[140, 1], requires_grad=0, device=cuda:0)):
  %4 : int = prim::Constant[value=1]()
  %5 : int = prim::Constant[value=1]()
  %6 : Float(1, 1, 140, strides=[140, 140, 1], requires_grad=0, device=cuda:0) = aten::unsqueeze(%3, %5)
  %7 : int = prim::Constant[value=1]()
  %8 : int = prim::Constant[value=1]()
  %9 : Float(1, 1, 140, strides=[140, 140, 1], requires_grad=0, device=cuda:0) = aten::unsqueeze(%2, %8)
  %10 : Tensor[] = prim::ListConstruct(%6, %9)
  %11 : int = prim::Constant[value=1]()
  %12 : int = prim::Constant[value=1]()
  %attention_weights_cat.2 : Float(1, 2, 140, strides=[280, 140, 1], requires_grad=0, device=cuda:0) = aten::cat(%10, %12)
  %14 : Tensor[] = prim::ListConstruct(%0, %1)
  %15 : int = prim::Constant[value=-1]()
  %cell_input.2 : Float(1, 768, strides=[768, 1], requires_grad=0, device=cuda:0) = aten::cat(%14, %15)
  return (%cell_input.2, %attention_weights_cat.2)
"""
g = torch._C.parse_ir(ir)

torch._C._jit_pass_fuse_tensorexprs(g)
print(g)
inputs = [torch.randn(1, 256, device='cuda'),
          torch.randn(1, 512, device='cuda'),
          torch.randn(1, 140, device='cuda'),
          torch.randn(1, 140, device='cuda')]
x = torch._C._jit_interpret_graph(g, tuple(inputs))