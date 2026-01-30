import weakref
import torch
import torch.autograd.forward_ad as fwAD
import gc

def scope():
  saved_tensors = []
  class A(torch.autograd.Function):
      @staticmethod
      def forward(x):
          return x

      @staticmethod
      def setup_context(ctx, inputs, output):
          ctx.mark_dirty(inputs[0])
          ctx.save_for_backward(output)
          saved_tensors.append(output)

      @staticmethod
      def backward(ctx, grad_output):
          return grad_output

      @staticmethod
      def jvp(ctx, x_t):
          x_t.add_(0)
          return x_t

  a = torch.tensor(2., device="cpu", requires_grad=True).clone()
  a_t = torch.tensor(2., device="cpu", requires_grad=True).clone()

  with fwAD.dual_level():
      a_dual = fwAD.make_dual(a, a_t)
      A.apply(a_dual)

  class Test():
      pass
  test = Test()
  ref = weakref.ref(test)
  saved_tensors[0].grad_fn.metadata["test"] = test

  return ref

ref = scope()
gc.collect()
print(ref())