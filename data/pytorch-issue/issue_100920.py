import torch

input_tensor = torch.randint(10, (3, 4, 5))
batch1 = torch.rand(3, 4, 6)
batch2 = torch.rand(3, 6, 5)

result = torch.ops.aten.baddbmm(input_tensor, batch1, batch2)
result_meta = torch.ops.aten.baddbmm(input_tensor.to("meta"), batch1.to("meta"), batch2.to("meta"))

print(result.dtype)      # torch.float32
print(result_meta.dtype) # torch.int64