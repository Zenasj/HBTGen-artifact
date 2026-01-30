import torch.nn as nn

import torch

torch.manual_seed(0)

input = torch.randint(0, 256, size=(1, 3, 256, 256), dtype=torch.uint8).contiguous(memory_format=torch.channels_last)
input = input[0]
input = input[None, ...]

assert input.is_contiguous(memory_format=torch.channels_last)

output = torch.nn.functional.interpolate(input, (224, 224), mode="bilinear", antialias=True)
expected = torch.nn.functional.interpolate(input.float(), (224, 224), mode="bilinear", antialias=True)

assert output.is_contiguous()
assert expected.is_contiguous()

torch.testing.assert_close(expected, output.float(), atol=1, rtol=1)
# > 
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#   File "/pytorch/torch/testing/_comparison.py", line 1511, in assert_close
#     raise error_metas[0].to_error(msg)
# AssertionError: Tensor-likes are not close!
#
# Mismatched elements: 14120 / 150528 (9.4%)
# Greatest absolute difference: 214.6112518310547 at index (0, 1, 152, 13) (up to 1 allowed)
# Greatest relative difference: 17.005144119262695 at index (0, 2, 26, 2) (up to 1 allowed)