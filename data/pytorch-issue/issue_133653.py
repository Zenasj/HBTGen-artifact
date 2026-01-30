import torchvision

# example.py
import torch
from torch.export import Dim, export
from torchvision.models import resnet50


def main():
    model = resnet50()
    example_args = (torch.randn(1, 3, 224, 224),)
    batch = Dim("batch", min=1, max=10)
    dynamic_shapes = {
        "x": {0: batch}
    }
    export(model, example_args, dynamic_shapes=dynamic_shapes)


if __name__ == "__main__":
    main()