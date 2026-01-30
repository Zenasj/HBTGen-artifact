import torch

results = {}
arg_1_tensor = torch.rand([512, 3, 144, 144], dtype=torch.float64)
arg_1 = arg_1_tensor.clone().to_sparse()  # Convert tensor to sparse

# CPU computation
try:
    results["res_cpu"] = torch.sparse.sum(arg_1)
except Exception as e:
    print("Error:" + str(e))

# GPU computation
arg_1_gpu = arg_1_tensor.clone().cuda().to_sparse()  # Convert tensor to sparse and move to GPU
try:
    results["res_gpu"] = torch.sparse.sum(arg_1_gpu)
except Exception as e:
    print("Error:" + str(e))

print(results)