import torch

@torch.jit.script
def foo(x, tup):
  # type: (int, Tuple[Tensor, Tensor]) -> Tensor
  t0, t1 = tup
  return t0 + t1 + x

foo.save("test.pt")

@torch.jit.script
def test_jit(input_fs1:Tuple[str]) -> Tuple[str]:
    res = ""
    for i in range(len(input_fs1)):
        res += input_fs1[i]
    return (res)

@torch.jit.script
def foo(x:int, tup:Tuple[int, int])->int:
    t0, t1 = tup
    return t0 + t1 + x

print(foo(3,(3,3)))
torch.jit.save(foo, "test.pt")