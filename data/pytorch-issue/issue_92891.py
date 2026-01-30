import torch
dtypes = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]

for dtype in dtypes:
    x = torch.empty(10000, dtype=dtype).exponential_() # should fail !
    print("dtype: ", x.dtype, "sum: ", x.sum())