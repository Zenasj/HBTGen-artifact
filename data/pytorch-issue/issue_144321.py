import numpy as np
import torch
import torch.nn as nn

# torch.rand(B, 256, 1, 1, dtype=torch.float16)

class Dense(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features=in_features, out_features=out_features,
                         bias=False, device="cpu", dtype=torch.float16)
        self.weight.requires_grad = False

class MyModel(nn.Module):
    def __init__(self, in_features=256, out_features=1024):
        super().__init__()
        self.pytorch_linear = Dense(in_features, out_features)

    def forward(self, x):
        # Flatten input to 2D (batch, features)
        x_2d = x.view(x.size(0), -1)
        pt_output = self.pytorch_linear(x_2d)

        # Compute numpy outputs
        input_np = x_2d.detach().cpu().numpy().astype(np.float32)
        weights_np = self.pytorch_linear.weight.detach().cpu().numpy().astype(np.float32)

        # NumPy implementation (matmul)
        np_output = numpy_dense(input_np, weights_np)
        np_output_tensor = torch.from_numpy(np_output.astype(np.float16)).to(pt_output)

        # NumPy from scratch implementation (explicit loops)
        np_from_scratch = numpy_dense_from_scratch(input_np, weights_np)
        np_from_scratch_tensor = torch.from_numpy(np_from_scratch.astype(np.float16)).to(pt_output)

        # Compare using allclose with float16 tolerances
        close_np = torch.allclose(pt_output, np_output_tensor, rtol=1e-3, atol=1e-4)
        close_from_scratch = torch.allclose(pt_output, np_from_scratch_tensor, rtol=1e-3, atol=1e-4)

        return torch.tensor([close_np and close_from_scratch], dtype=torch.bool)

def numpy_dense(x, weights):
    x = x.astype(np.float32)
    weights = weights.astype(np.float32)
    res = np.matmul(x, weights.T, dtype=np.float32)
    return res.astype(np.float16)

def numpy_dense_from_scratch(x, weights):
    x_fp32 = x.astype(np.float32)
    weights_fp32 = weights.astype(np.float32)
    result_fp16 = np.zeros((x.shape[0], weights.shape[0]), dtype=np.float16)
    for i in range(x_fp32.shape[0]):
        for j in range(weights_fp32.shape[0]):
            sum_fp32 = 0.0
            for k in range(x_fp32.shape[1]):
                sum_fp32 += x_fp32[i, k] * weights_fp32[j, k]
            result_fp16[i, j] = np.float16(sum_fp32)
    return result_fp16

def my_model_function():
    return MyModel(in_features=256, out_features=1024)

def GetInput():
    B = 64  # Inferred from original issue's input size (64,256)
    return torch.rand(B, 256, 1, 1, dtype=torch.float16)

