import torch

graph = torch.parse_ir(
    """
graph(%x: Tensor):
  %type: int = prim::Constant[value=1]()
{}
  %ret: float[] = prim::tolist(%x, %dim, %type)
  return (%ret)
""".format(
        "  %dim: int = aten::dim(%x)\n" * 100_000
    )
)

x = torch.randn(4)
res = torch._C._jit_interpret_graph(graph, (x,))
print(res)