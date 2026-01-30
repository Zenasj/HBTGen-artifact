import numpy as np
import torch
import torch.nn as nn
import argparse


def numpy_dense_from_scratch(x, weights):
    # Convert to float32 for accumulation
    x_fp32 = x.astype(np.float32)
    weights_fp32 = weights.astype(np.float32)

    result_fp16 = np.zeros((x.shape[0], weights.shape[0]), dtype=np.float16)
    for i in range(x_fp32.shape[0]):
        for j in range(weights_fp32.shape[0]):
            # Accumulate the result in float32 for better precision
            sum_fp32 = 0.0
            for k in range(x_fp32.shape[1]):
                sum_fp32 += x_fp32[i, k] * weights_fp32[j, k]
            # Store the final result in float16 after accumulation
            result_fp16[i, j] = np.float16(sum_fp32)

    return result_fp16


def numpy_dense(x, weights):
    x = x.astype(np.float32)
    weights = weights.astype(np.float32)
    res = np.matmul(x, weights.T, dtype=np.float32)
    return res.astype(np.float16)


class Dense(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features=in_features, out_features=out_features,
                         bias=False, device="cpu", dtype=torch.float16)
        self.weight.requires_grad = False

    def forward(self, input):
        return super().forward(input)


def compare_outputs(pytorch_model, inputs):
    def _to_numpy(tensor):
        return tensor.cpu().numpy()

    # Torch inference
    pytorch_model.eval()
    torch_outputs = [_to_numpy(pytorch_model(inputs))]

    # Numpy outputs
    numpy_outputs = [numpy_dense(_to_numpy(inputs), _to_numpy(pytorch_model.weight))]

    # Numpy from scratch outputs
    numpy_from_scratch_outputs = [numpy_dense_from_scratch(
        _to_numpy(inputs), _to_numpy(pytorch_model.weight))]

    # Both tests fail
    np.testing.assert_array_equal(torch_outputs, numpy_from_scratch_outputs)
    np.testing.assert_array_equal(torch_outputs, numpy_outputs)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--full_range", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(0)

    # Create random inputs either between [0, 1] or between [fp16min, fp16max]
    size = (64, 256)
    x_rand_tensor = torch.rand(size, requires_grad=False, dtype=torch.float32)
    f16_min = torch.finfo(torch.float16).min + 1
    f16_max = torch.finfo(torch.float16).max - 1

    # Inputs for test
    scale_factor = 1
    offset = 0
    if args.full_range:
        scale_factor = (f16_max - f16_min)
        offset = f16_min

    x = (x_rand_tensor * scale_factor + offset).to(torch.float16)

    # Create the model
    dense_model = Dense(256, 1024)
    compare_outputs(dense_model, x)


if __name__ == "__main__":
    main()