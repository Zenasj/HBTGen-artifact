import numpy as np
from numpy import vectorize, uint8
import torch as th
from torch import tensor, allclose
import cv2

def choose_colour (row_ix: uint8, col_ix: uint8, _channel_ix: uint8) -> uint8:
  if row_ix == 1 and col_ix == 1:
    return 127
  elif (row_ix == 2 and col_ix > 0) or (col_ix == 2 and row_ix > 0):
    return 255
  return 0

def get_tensor():
  img = np.fromfunction(function=vectorize(choose_colour), shape=(3, 3, 3), dtype=uint8)
  img = img.astype(np.float32)
  img = img / 255
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = th.from_numpy(np.transpose(img, (2, 0, 1))).float()
  return img

hardcoded_tensor = tensor(
  [[[0.0000, 0.0000, 0.0000],
    [0.0000, 0.4980, 1.0000],
    [0.0000, 1.0000, 1.0000]],

  [[0.0000, 0.0000, 0.0000],
    [0.0000, 0.4980, 1.0000],
    [0.0000, 1.0000, 1.0000]],

  [[0.0000, 0.0000, 0.0000],
    [0.0000, 0.4980, 1.0000],
    [0.0000, 1.0000, 1.0000]]])

unsqueezed_via_cpu = hardcoded_tensor.unsqueeze(0).to('cpu')
unsqueezed_via_mps = hardcoded_tensor.unsqueeze(0).to('mps').cpu()
assert allclose(unsqueezed_via_cpu, unsqueezed_via_mps, rtol=0.0001), "unsqueezing the harcoded tensor gives the same result on both CPU and on MPS"

computed_tensor = get_tensor()
print('hardcoded_tensor:\n', hardcoded_tensor)
print('computed_tensor:\n', computed_tensor)
assert allclose(computed_tensor, hardcoded_tensor, rtol=0.0001), "the hardcoded tensor is equivalent to the one we compute via get_tensor, so we should get the same result when we unsqueeze it… right?"

unsqueezed_via_cpu = computed_tensor.unsqueeze(0).to('cpu')
unsqueezed_via_mps = computed_tensor.unsqueeze(0).to('mps').cpu()
print('unsqueezed_via_cpu:\n', unsqueezed_via_cpu)
print('unsqueezed_via_mps:\n', unsqueezed_via_mps)
assert allclose(unsqueezed_via_cpu, unsqueezed_via_mps, rtol=0.0001), "unsqueezed MPS tensor differs from CPU counterpart"

import numpy as np
from numpy import float32, array
import torch as th
from torch import tensor

arr = array([[[0, 0],
        [0, 1]],

       [[0, 0],
        [0, 1]]], dtype=float32)

def get_tensor():
  arr_t = np.transpose(arr, (2, 0, 1))
  img = th.from_numpy(arr_t)
  img = img.unsqueeze(0)
  return img

hardcoded_tensor = tensor([[[[0., 0.],
          [0., 0.]],

         [[0., 1.],
          [0., 1.]]]])

via_mps = hardcoded_tensor.to('mps').cpu()
assert th.equal(hardcoded_tensor, via_mps), "hardcoded_tensor == {hardcoded_tensor -> GPU -> CPU}"

computed_tensor = get_tensor()
print('hardcoded_tensor:\n', hardcoded_tensor)
print('computed_tensor:\n', computed_tensor)
assert th.equal(computed_tensor, hardcoded_tensor), "hardcoded_tensor == get_tensor()"

via_mps = computed_tensor.to('mps').cpu()
print('computed_on_cpu:\n', computed_tensor)
print('computed_via_mps:\n', via_mps)
assert th.equal(computed_tensor, via_mps), "get_tensor() == {get_tensor() -> GPU -> CPU}" # fails!

torch.equal(hardcoded_tensor, computed_tensor)
True

hardcoded_tensor
tensor([[[[0., 0.],
          [0., 0.]],

         [[0., 1.],
          [0., 1.]]]])

computed_tensor
tensor([[[[0., 0.],
          [0., 0.]],

         [[0., 1.],
          [0., 1.]]]])

hardcoded_tensor.to('mps')
tensor([[[[0., 0.],
          [0., 0.]],

         [[0., 1.],
          [0., 1.]]]], device='mps:0')

computed_tensor.to('mps')
tensor([[[[0., 0.],
          [0., 0.]],

         [[0., 0.],
          [0., 0.]]]], device='mps:0')

def test_contiguous_minimal():
    device = "mps"

    x = torch.randn((1, 3, 2), dtype=torch.float)
    print("Is view or contiguous:", x._is_view(), x.is_contiguous(), x.stride())  # False True (6, 2, 1)

    x_view = x.squeeze(0)
    print("Is view or contiguous:", x_view._is_view(), x_view.is_contiguous(), x_view.stride())  # True True (2, 1)

    x_mps = x_view.to(device)
    print("Is view or contiguous:", x_mps._is_view(), x_mps.is_contiguous(), x_mps.stride())  # False True (2, 1)
    
    torch.testing.assert_close(x_mps, x_view, check_device=False)  # pass

    x = torch.randn((3, 2), dtype=torch.float)
    print("Is view or contiguous:", x._is_view(), x.is_contiguous(), x.stride())  # False True (2, 1)

    x_view = x.permute((1, 0))
    print("Is view or contiguous:", x_view._is_view(), x_view.is_contiguous(), x_view.stride())  # True False (1, 2)

    x_mps = x_view.to(device)
    print("Is view or contiguous:", x_mps._is_view(), x_mps.is_contiguous(), x_mps.stride())  # False False (1, 2)
    
    torch.testing.assert_close(x_mps, x_view, check_device=False)  # fail