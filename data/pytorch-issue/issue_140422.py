import torch

jit_graph = torch._C.parse_ir(
    """
    graph():
      %1 : Long(requires_grad=0, device=cpu) = prim::Constant[value={1}]()
      return (%1)
    """,
    parse_tensor_constants=True,
)
print(jit_graph)