# torch.rand(B, C, H, W, dtype=torch.uint8)
import torch
import torch.nn as nn

@torch.jit.script
def center_slice_helper(x, h_offset, w_offset, h_end, w_end):
    return x[:, :, h_offset:h_end, w_offset:w_end]

class MyModel(nn.Module):
    def __init__(self, crop_size):
        super(MyModel, self).__init__()
        self.crop_size = crop_size
        self.register_buffer('crop_size_t', torch.tensor(crop_size))

    def extra_repr(self):
        return 'crop_size={}'.format(self.crop_size)

    def forward(self, x):
        height, width = x.shape[2], x.shape[3]
        if not isinstance(height, torch.Tensor):
            height, width = torch.tensor(height).to(x.device), torch.tensor(width).to(x.device)
        h_offset = (height - self.crop_size_t) / 2
        w_offset = (width - self.crop_size_t) / 2
        h_end = h_offset + self.crop_size_t
        w_end = w_offset + self.crop_size_t
        return center_slice_helper(x, h_offset.long(), w_offset.long(), h_end.long(), w_end.long())

def my_model_function():
    return MyModel(224)

def GetInput():
    return torch.randn(1, 3, 300, 256, device='cpu').byte()

