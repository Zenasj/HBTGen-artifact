import torch.nn as nn

import torch
                                                                                                    
input = torch.randn([1, 1, 1, 1, 1])
input = torch.quantize_per_tensor(input, 0.1, 10, torch.qint32)
torch.quantized_max_pool3d(input, (1, 1, 1), (1, 1, 1), (0, 0, 0), (-3, 1, 1)) # crash
                                                                                                    
input = torch.randn([1, 1, 1, 1, 1])
input = torch.quantize_per_tensor(input, 0.1, 10, torch.qint32)
result = torch.nn.functional.max_pool3d(input, (1, 1, 1), (1, 1, 1), (0, 0, 0), (-3, 1, 1))  # crash