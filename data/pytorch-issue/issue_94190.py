import torch
shape=[3, 2, 2]

content_im = (torch.randn(shape).float())
content_im = content_im.permute((2, 0, 1))
content_im = content_im.to('mps')

content_im *1.0  # an operation on tensor is required to trigger the issue, it can be .sum() as well

print('zero_like on .contiguous()')
print(torch.zeros_like(content_im.contiguous()))

print('zero_like on permuted version') # this fails
print(torch.zeros_like(content_im))     # it output the content of content_im

self.assertEqual(torch.zeros_like(cpu_out), torch.zeros_like(mps_out))