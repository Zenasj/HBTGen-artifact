import torch
import torch.nn as nn


def torch_memory(device):
    # Checks and prints GPU memory
    print(f'{torch.cuda.memory_allocated(device)/1024/1024:.2f} MB USED')
    print(f'{torch.cuda.memory_reserved(device)/1024/1024:.2f} MB RESERVED')
    print(f'{torch.cuda.max_memory_allocated(device)/1024/1024:.2f} MB USED MAX')
    print(f'{torch.cuda.max_memory_reserved(device)/1024/1024:.2f} MB RESERVED MAX')
    print('')


print(torch.__version__)

device = 0
device = torch.device(device)

# Create input
x = torch.randn((1, 2, 24, 512, 512), dtype=torch.float32, device=device)

# Create conv layers
pmode = 'circular'
# pmode = 'zeros'
width = 64
conv0 = torch.nn.Conv3d(2, width, 3, padding=1, padding_mode=pmode).to(device)
conv1 = torch.nn.Conv3d(width, width, 3, padding=1, padding_mode=pmode).to(device)
conv2 = torch.nn.Conv3d(width, 2, 3, padding=1, padding_mode=pmode).to(device)
torch_memory(device.index)

with torch.no_grad():
    x = conv0(x)
    print('After first conv layer.')
    torch_memory(device.index)

    x = conv1(x)
    print('After second conv layer.')
    torch_memory(device.index)  # A lot more memory is reserved here

    x = conv2(x)
    print('After third conv layer.')
    torch_memory(device.index)

def _pad_circular(input, padding):
    # type: (Tensor, List[int]) -> Tensor
    """
    Args:
        input: Tensor that follows the formatting of the input to convolution
            layers.
        padding: Tuple with length two times the degree of the convolution. The
            order of the integers in the tuple are shown in the following
            example:

            For 3D convolutions:
                padding[-2] is the amount of padding applied to the beginning
                    of the depth dimension.
                padding[-1] is the amount of padding applied to the end of the
                    depth dimension.
                padding[-4] is the amount of padding applied to the beginning
                    of the height dimension.
                padding[-3] is the amount of padding applied to the end of the
                    height dimension.
                padding[-6] is the amount of padding applied to the beginning
                    of the width dimension.
                padding[-5] is the amount of padding applied to the end of the
                    width dimension.

    Returns:
        out: Tensor with padded shape.
    """
    shape = input.shape
    ndim = len(shape[2:])

    # Only supports wrapping around once
    for a, size in enumerate(shape[2:]):
        assert padding[-(a*2+1)] <= size
        assert padding[-(a*2+2)] <= size

    # Get shape of padded array
    new_shape = shape[:2]
    for a, size in enumerate(shape[2:]):
        new_shape += (size + padding[-(a*2+1)] + padding[-(a*2+2)],)

    out = torch.empty(new_shape, dtype=input.dtype, layout=input.layout,
                      device=input.device)

    # Put original array in padded array
    if ndim == 1:
        out[..., padding[-2]:-padding[-1]] = input
    elif ndim == 2:
        out[..., padding[-2]:-padding[-1], padding[-4]:-padding[-3]] = input
    elif ndim == 3:
        out[..., padding[-2]:-padding[-1], padding[-4]:-padding[-3], padding[-6]:-padding[-5]] = input

    # Pad right side, then left side.
    # Corners will be written more than once when ndim > 1

    # Pad first conv dim
    out[:, :, :padding[-2]] = out[:, :, -(padding[-2] + padding[-1]):-padding[-1]]
    out[:, :, -padding[-1]:] = out[:, :, padding[-2]:(padding[-2] + padding[-1])]

    if len(padding) > 2:
        # Pad second conv dim
        out[:, :, :, :padding[-4]] = out[:, :, :, -(padding[-4] + padding[-3]):-padding[-3]]
        out[:, :, :, -padding[-3]:] = out[:, :, :, padding[-4]:(padding[-4] + padding[-3])]

    if len(padding) > 4:
        # Pad third conv dim
        out[:, :, :, :, :padding[-6]] = out[:, :, :, :, -(padding[-6] + padding[-5]):-padding[-5]]
        out[:, :, :, :, -padding[-5]:] = out[:, :, :, :, padding[-6]:(padding[-6] + padding[-5])]

    return out