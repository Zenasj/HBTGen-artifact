import torch

if __name__ == '__main__':
    ir = """
graph(%x : Tensor,
      %y : Tensor):
  %2 : float = prim::Constant[value=1.2]()
  %result : Tensor= aten::add(%x, %2, %y)
  return (%result)
"""
    x = torch.tensor([[1., 2.], [3., 4.]])
    y = torch.tensor([[2., 1.], [2., 1.]])
    graph = torch._C.parse_ir(ir)
    print(graph)
    graph.alias_db().analyze()
    # print(script(x, y))