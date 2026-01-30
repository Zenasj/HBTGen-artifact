import torch

torch._C.parse_ir(
    """
graph():
  %0 : float = prim::Constant[value=-0.31622776601683789]()
  %1 : float = prim::Constant[value=0.31622776601683789]()
  %2 : Generator = prim::Constant[value=torch.Generator(device="cpu", seed=352461024221769975)]()
  %3 : NoneType = prim::Constant()
  %4 : int[] = prim::Constant[value=[10, 10]]()
  %5 : int = prim::Constant[value=6]()
  %6 : Device = prim::Constant[value="cpu"]()
  %7 : Tensor = aten::empty(%4, %5, %3, %6, %3, %3)
  %8 : Float(10, 10) = aten::uniform(%7, %0, %1, %2)
  return (%8)
    """,
)