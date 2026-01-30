[[torch.uint8, torch.int8], [torch.uint8, torch.int16], [torch.uint8, torch.int32],
       [torch.uint8, torch.int64], [torch.uint8, torch.float64], [torch.uint8, torch.float32], [torch.uint8, torch.float16]]

[[torch.uint8, torch.int8], [torch.uint8, torch.int16], [torch.uint8, torch.int32],
       [torch.uint8, torch.int64], [torch.uint8, torch.float64], [torch.uint8, torch.float16], [torch.uint8, torch.float32]]

import torch
import numpy
x = torch.tensor([[66, 63, 21, 84, 86, 90, 86, 21],
                            [81, 68, 22, 39, 43, 38, 24, 63],
                            [96, 98, 33, 85, 90, 87, 52, 99],
                            [56, 21, 95, 59, 85, 51, 16, 37],
                            [74, 55, 95, 97, 45, 82, 71, 33],
                            [88, 90, 91, 68, 31, 22, 53, 30],
                            [68, 22, 73, 82, 36, 56, 96, 20],
                            [71, 66, 82, 17, 97, 35, 20, 43],
                            [31, 81, 87, 90, 60, 49, 96, 91]], dtype=torch.uint8)
y = torch.nansum(x, dtype=torch.float16)
z = numpy.nansum(x.detach().cpu().numpy(), dtype=numpy.float16)
# y is 4444.0, but z is 4450.0

x = torch.tensor([[66, 63, 21, 84, 86, 90, 86, 21],
                  [81, 68, 22, 39, 43, 38, 24, 63],
                  [96, 98, 33, 85, 90, 87, 52, 99],
                  [56, 21, 95, 59, 85, 51, 16, 37],
                  [74, 55, 95, 97, 45, 82, 71, 33],
                  [88, 90, 91, 68, 31, 22, 53, 30],
                  [68, 22, 73, 82, 36, 56, 96, 20],
                  [71, 66, 82, 17, 97, 35, 20, 43],
                  [31, 81, 87, 90, 60, 49, 96, 91]], dtype=torch.int16)

print(torch.sum(x, dtype=torch.float16))