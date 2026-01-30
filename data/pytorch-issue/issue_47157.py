import torch
names = ["batch", "height", "width", "complex"]
z = torch.ones((5, 12, 14, 2), requires_grad=True).refine_names(*names)
z_complex = torch.view_as_complex(z.rename(None)).refine_names(*names[:-1])

def view_as_complex(data):
    """Named version of `torch.view_as_complex()`"""
    data = data.clone()
    real_part = data[..., 0]
    imag_part = data[..., 1]

    return (real_part + 1j*imag_part).refine_names(*data.names[:-1])