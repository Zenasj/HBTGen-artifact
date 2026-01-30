import torch
import timeit
timeit.timeit('torch.randn(10000).repeat_interleave(100, -1)', 'import torch', number=100)
timeit.timeit('torch.randn(10000)[..., None].expand(-1, 100).flatten(-2, -1)', 'import torch', number=100)

0.37211750000000166
0.10460659999995414

self.erb_indices = torch.tensor([
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 7, 7, 8, 
    10, 12, 13, 15, 18, 20, 24, 28, 31, 37, 42, 50, 56, 67
])
gains = torch.randn(32)

gains = torch.repeat_interleave(gains, self.erb_indices)

gains = torch.cat([x.repeat(num_x) for x, num_x in zip(gains, self.erb_indices)])