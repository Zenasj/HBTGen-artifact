import torch

TEST_LARGE_TENSOR = torch.cuda.get_device_properties(0).total_memory >= 12e9

TEST_LARGE_TENSOR = torch.cuda.get_device_properties(0).total_memory >= 12e9