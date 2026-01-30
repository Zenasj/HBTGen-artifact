import torch.nn as nn

import torch

def _interpolate_wrapper(input_tensor: torch.Tensor) -> torch.Tensor:
    print(f'run interpolation with input shape {input_tensor.shape}')
    return torch.nn.functional.interpolate(
        input_tensor,
        scale_factor=.5,
        antialias=True,
        mode='bilinear',
        align_corners=False
    )

# create test data
input_tensor_3D = torch.rand([3, 800, 600])
input_tensor_4D = input_tensor_3D.unsqueeze(0)

# interpolate a 4D tensor
_interpolate_wrapper(input_tensor_4D)  # this runs fine
print('====interpolation of a 4D tensor run successfully')

# interpolate a 3D tensor
_interpolate_wrapper(input_tensor_3D)  # this triggers the error