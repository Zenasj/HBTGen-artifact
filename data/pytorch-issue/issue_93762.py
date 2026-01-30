import torch

def test_tree_map(self):
          def f(x):
              if isinstance(x, torch.Tensor):
                  return x.abs()
              else:
                  return x
  
          @torchdynamo.optimize("eager", nopython=True)
          def fn(x):
              return torch.utils._pytree.tree_flatten((x, x))
  
          fn(torch.randn(3))