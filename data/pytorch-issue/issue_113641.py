import torch


def eager(input, top, left, height, width):
    output = input - torch.tensor([left, top, left, top], dtype=input.dtype, device=input.device)
    output[..., 0::2].clamp_(min=0, max=width)
    output[..., 1::2].clamp_(min=0, max=height)
    return output


compiled_static = torch.compile(eager, dynamic=False)
compiled_dynamic = torch.compile(eager, dynamic=True)

input = torch.tensor([[0.0, 1.0, 10.0, 14.0]])
kwargs = dict(top=7, left=3, height=3, width=5)

print("eager", eager(input, **kwargs))
print("compiled static", compiled_static(input, **kwargs))
print("compiled dynamic", compiled_dynamic(input, **kwargs))