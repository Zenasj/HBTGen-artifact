import torch.nn as nn

import torch
import torch.onnx

class MyModel(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        r = torch.tensor([0, 0])

        if x.numel() > 0:
            if y.numel() > 0:
                return r + 1, r + 1

        return r, r


instance = MyModel()
instance.eval()

x, y = torch.tensor([1, 2, 3, 4]), torch.tensor([1, 2, 3, 4, 5])

torch_result = instance(x, y)

scripted_cell = torch.jit.script(instance)

torch.onnx.export(
    model=scripted_cell,
    args=(x, y),
    f='net.onnx',
    verbose=True,
    input_names=['x', 'y'],
    output_names=['result', 'result2'],
    example_outputs=torch_result,
    opset_version=13,
    dynamic_axes={'y': {0: 'first_axis'}},
)

def forward(self,
    x: Tensor,
    y: Tensor) -> Tuple[Tensor, Tensor]:
  _0 = uninitialized(Tuple[Tensor, Tensor])  # <<<< this one!
  r = torch.tensor([0, 0])
  if torch.gt(torch.numel(x), 0):
    if torch.gt(torch.numel(y), 0):
      _5 = (torch.add(r, 1), torch.add(r, 1))
      _3, _4 = True, _5
    else:
      _3, _4 = False, _0
    _1, _2 = _3, _4
  else:
    _1, _2 = False, _0
  if _1:
    _6 = _2
  else:
    _6 = (r, r)
  return _6