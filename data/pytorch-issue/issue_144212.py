import torch.nn as nn

import torch
import onnxruntime as ort
import onnx
import torch.onnx
import numpy as np
import time
import ctypes
import onnxruntime as ort
# export LD_LIBRARY_PATH=/home/vipuser/miniconda3/envs/torch/lib/python3.12/site-packages/nvidia/cudnn/lib/:$LD_CLIBRARY_PATH

class Averager:
    def __init__(self):
        self.call_count = {}
        self.total_sum = {}    

    def __call__(self, name, num):
        if name not in self.call_count:
            self.call_count[name] = 0
        if name not in self.total_sum:
            self.total_sum[name] = 0
            # ignore first time
            return
        self.call_count[name] += 1                # Increment call count
        self.total_sum[name] += num            # Add result to total sum
    
    def get(self, name):
        assert name in self.call_count
        assert name in self.total_sum
        return self.total_sum[name]/self.call_count[name]


class Cdist(torch.nn.Module):
    def forward(self, x, y, p=2.0, compute_mode="use_mm_for_euclid_dist_if_necessary"):
        # type: (Tensor, Tensor, float, str) -> (Tensor)
        r"""Computes batched the p-norm distance between each pair of the two collections of row vectors.

        Args:
            x1 (Tensor): input tensor of shape :math:`B \times P \times M`.
            x2 (Tensor): input tensor of shape :math:`B \times R \times M`.
            p: p value for the p-norm distance to calculate between each vector pair
                :math:`\in [0, \infty]`.
            compute_mode:
                'use_mm_for_euclid_dist_if_necessary' - will use matrix multiplication approach to calculate
                euclidean distance (p = 2) if P > 25 or R > 25
                'use_mm_for_euclid_dist' - will always use matrix multiplication approach to calculate
                euclidean distance (p = 2)
                'donot_use_mm_for_euclid_dist' - will never use matrix multiplication approach to calculate
                euclidean distance (p = 2)
                Default: use_mm_for_euclid_dist_if_necessary.

        If x1 has shape :math:`B \times P \times M` and x2 has shape :math:`B \times R \times M` then the
        output will have shape :math:`B \times P \times R`.

        This function is equivalent to `scipy.spatial.distance.cdist(input,'minkowski', p=p)`
        if :math:`p \in (0, \infty)`. When :math:`p = 0` it is equivalent to
        `scipy.spatial.distance.cdist(input, 'hamming') * M`. When :math:`p = \infty`, the closest
        scipy function is `scipy.spatial.distance.cdist(xn, lambda x, y: np.abs(x - y).max())`.

        """
        return torch.cdist(x, y, p, compute_mode)
    

def test(compute_mode = "use_mm_for_euclid_dist", p=2.0):
    print(">>", compute_mode, p)
    model = Cdist()  # Instantiate the model
    # Export the model to ONNX format
    x = torch.randn(1, 640, 128, ) 
    y = torch.randn(2, 1280, 128, )

    torch.onnx.export(model,
                    (x, y, p, compute_mode),  # Pass the two dummy inputs as a tuple
                    "cdist_.onnx",  # Output ONNX file
                    input_names=["x", "y",],  # Names for the input tensors
                    output_names=["output"],  # Name for the output tensor
                    #   dynamic_axes={"input1": {0: "batch_size"}, "input2": {0: "batch_size"}, "output": {0: "batch_size"}},  # Optional: dynamic batch size
                    opset_version=9)  # Specify the ONNX opset version

    inputs = [x, y]
    names = ["x", "y"]
    inputs_dict = {i:j for i, j in zip(names, inputs)}
    
    start = time.time()
    output_pytorch = model(x.cuda(), y.cuda(), p, compute_mode).cpu()
    # print(output_pytorch.size(), output_pytorch[0][0][:3])
    # print("torch model run for:", time.time()-start, "s")
    

    inputs_dict = {i:j.numpy() for i, j in zip(names, inputs)}
    onnx_model_path = "cdist_.onnx"  # Replace with the path to your ONNX model
    session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])
    start = time.time()
    onnx_output = session.run(["output"], inputs_dict)[0]
    onnx_time_cost = time.time()-start
    # print("onnx model run for:", onnx_time_cost, "s")
    onnx_output_tensor = torch.from_numpy(onnx_output)
    # print(onnx_output_tensor.size(), onnx_output_tensor[0][0][:3])
    if not torch.allclose(output_pytorch, onnx_output_tensor, rtol=1e-2):
        print("The outputs of the PyTorch model and the ONNX model are different!")
    else:
        print("ok")
    return onnx_time_cost

if __name__=="__main__":
    avg = Averager()
    testtimes = 20
    p = 2.0
    for i in range(testtimes):
        avg("donot_use_mm_for_euclid_dist", test("donot_use_mm_for_euclid_dist", p))
        avg("use_mm_for_euclid_dist_if_necessary", test("use_mm_for_euclid_dist_if_necessary", p))
        avg("use_mm_for_euclid_dist", test("use_mm_for_euclid_dist", p))
    print("donot_use_mm_for_euclid_dist: average time:", avg.get("donot_use_mm_for_euclid_dist"))
    print("use_mm_for_euclid_dist_if_necessary: average time:", avg.get("use_mm_for_euclid_dist_if_necessary"))
    print("use_mm_for_euclid_dist: average time:", avg.get("use_mm_for_euclid_dist"))