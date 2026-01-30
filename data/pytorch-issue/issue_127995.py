py
import torch
import numpy as np # E: module level import not at top of file # E: at least two spaces before inlin

lib = torch.library.Library("mylib", "FRAGMENT") 
lib.define("numpy_sin(Tensor input, Tensor(a!) output) -> ()")

def numpy_sin(input: torch.Tensor, output: torch.Tensor) -> None:
    assert input.device == output.device
    assert input.device.type == "cpu"
    input_np = input.numpy()
    output_np = output.numpy()
    np.sin(input_np, out=output_np)

lib.impl("numpy_sin", numpy_sin, "CPU")

numpy_sin = torch.ops.mylib.numpy_sin

@torch.compile(fullgraph=True)
def f(x):
    out = torch.empty(3)
    numpy_sin(x, out)
    return out

x = torch.randn(3)
y = f(x)
print(torch.__version__)
assert torch.allclose(y, x.sin())

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (3, ), (1, ))
    buf0 = empty_strided_cpu((3, ), (1, ), torch.float32)
    # Source Nodes: [], Original ATen: []
    buf1 = torch.ops.mylib.numpy_sin.default(arg0_1, buf0)
    del arg0_1
    cpp_fused_empty_0(buf0)
    return (buf0, )

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (3, ), (1, ))
    buf0 = empty_strided_cpu((3, ), (1, ), torch.float32)
    # Source Nodes: [], Original ATen: []
    buf1 = torch.ops.mylib.numpy_sin.default(arg0_1, buf0)
    del arg0_1
    return (buf0, )

def call(args):
      arg0_1, = args
      args.clear()
      assert_size_stride(arg0_1, (3, ), (1, ))
      buf0 = empty_strided_cpu((3, ), (1, ), torch.float32)
      cpp_fused_empty_0(buf0)
      # Source Nodes: [], Original ATen: []
      buf1 = torch.ops.mylib.numpy_sin.default(arg0_1, buf0)
      del arg0_1
      return (buf0, )

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (3, ), (1, ))
    buf0 = empty_strided_cpu((3, ), (1, ), torch.float32)
    # Source Nodes: [], Original ATen: []
    buf1 = torch.ops.mylib.numpy_sin.default(arg0_1, buf0)
    del arg0_1
    cpp_fused_empty_0(buf0)
    return (buf0, )