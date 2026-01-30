import torch
import torchvision

import timeit

if __name__ == '__main__':
    # model = Net()
    # input = torch.rand(1, 1, 28, 28)
    model = torchvision.models.resnet50(pretrained=True).eval()
    input = torch.rand(1, 3, 224, 224)

    print("torch version:", torch.__version__)
    print("torchvision version:", torchvision.__version__)

    print("no shape inferece:",
          timeit.timeit(lambda: torch.onnx._export(model, (input,), "test.onnx", onnx_shape_inference=False),
                        number=10))
    print("shape inference",
          timeit.timeit(lambda: torch.onnx._export(model, (input,), "test.onnx", onnx_shape_inference=True),
                        number=10))