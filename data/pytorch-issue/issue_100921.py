import torch

grad = torch.randint(10, (4,))
input_tensor = torch.randint(10, (4,))

result = torch.ops.aten.threshold_backward(grad, input_tensor, 1)
result_meta = torch.ops.aten.threshold_backward(grad.to("meta"), input_tensor.to("meta"), 1)

print(result.dtype)      # torch.int64
print(result_meta.dtype) # torch.float32