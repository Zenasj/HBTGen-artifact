import torch
from torch._dynamo.utils import is_compile_supported

if not torch.cuda.is_available():
    exit()

result_cuda = is_compile_supported("cuda")
result_cuda0 = is_compile_supported("cuda:0")

print("result_cuda:", result_cuda)
print("result_cuda0:", result_cuda0)