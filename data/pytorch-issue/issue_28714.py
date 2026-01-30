import torch

x = torch.randn([1, 1, 16777217, 2])                                                                                                                                                                                                                   
input = x.cuda().contiguous(memory_format=torch.channels_last)                                                                                                                                                                                         
kernel_size, stride, padding, dilation, ceil_mode = 1, 1, 0, 1, False                                                                                                                                                                                 
torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)