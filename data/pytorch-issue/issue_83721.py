import torch.nn as nn

from typing import List, Dict
import torch

x = torch.tensor([[59, 26, 32, 31, 58, 37, 12,  8,  8, 32, 27, 27, 35,  9,  3, 44, 22, 36,
                   22, 61, 51, 35, 15, 13, 14, 32, 22, 21,  9]], dtype=torch.long)

nums = [3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27,
        28, 29, 30, 31, 37, 38, 39, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]


@torch.jit.script
def batch(x, l: List[int]):
    for i in range(len(x)):
        for j in range(len(x[i])):
            if x[i, j] in l:
                x[i, j] *= 2
    return x


class Module1(torch.nn.Module):
    def forward(self, x):
        return batch(x, nums)


m1 = Module1()
print(m1(x))

torch.onnx.export(m1,
                  (x),
                  "2.onnx",
                  verbose=True,
                  input_names=["x"],
                  dynamic_axes={
                      "x": {
                          1: "frames",
                      },
                  },
                  opset_version=11,
                  )

@torch.jit.script
def batch(x, l: List[int]):
    for i in range(len(x)):
        for j in range(len(x[i])):
            if x[i, j] in l:
                x[i, j] *= 2
    return x