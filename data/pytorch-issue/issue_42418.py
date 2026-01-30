import torch
print('torch.__version__', torch.__version__)

x = torch.tensor([[1., 2, 3], [4., 5, 6]]).to(0)

# Do segment_axis
# https://github.com/fgnt/nara_wpe/blob/ae0fd6444a6bb03aa6e29ea2b698e6b73d596253/nara_wpe/wpe.py#L14
# This is a view on the input data, but with a segmentation that is used for example in the stft.
# This view can save large memory.
stride = list(x.stride())
stride.insert(1, 1)
shape = list(x.size())
shape.insert(1, 2)
shape[-1] -= 1
c = torch.as_strided(x, size=shape, stride=stride)

print(c.stride(), c.shape)

torch.einsum('...ab,...bc->...ac', c, c)