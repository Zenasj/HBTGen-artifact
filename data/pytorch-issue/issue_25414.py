import torchvision

@torch.jit.script
def center_slice_helper(x, h_offset, w_offset, h_end, w_end):
    return x[:, :, h_offset:h_end, w_offset:w_end]


class CenterCrop(nn.Module):
    def __init__(self, crop_size):
        """Crop from the center of a 4d tensor
        Input shape can be dynamic
        :param crop_size: the center crop size
        """
        super(CenterCrop, self).__init__()
        self.crop_size = crop_size

    def extra_repr(self):
        """Extra information
        """
        return 'crop_size={}'.format(
            self.crop_size
        )

    def forward(self, x):
        h_offset = (x.shape[2] - self.crop_size) / 2
        w_offset = (x.shape[3] - self.crop_size) / 2
        if not isinstance(h_offset, torch.Tensor):
            h_offset, w_offset = torch.tensor(h_offset), torch.tensor(w_offset)
        h_end = h_offset + self.crop_size
        w_end = w_offset + self.crop_size
        return center_slice_helper(x, h_offset, w_offset, h_end, w_end)

import torch.nn as nn

@torch.jit.script
def center_slice_helper(x, h_offset, w_offset, h_end, w_end):
    return x[:, :, h_offset:h_end, w_offset:w_end]


class CenterCrop(nn.Module):
    def __init__(self, crop_size):
        """Crop from the center of a 4d tensor
        Input shape can be dynamic
        :param crop_size: the center crop size
        """
        super(CenterCrop, self).__init__()
        self.crop_size = crop_size

    def extra_repr(self):
        """Extra information
        """
        return 'crop_size={}'.format(
            self.crop_size
        )

    def forward(self, x):
        h_offset = (x.shape[2] - self.crop_size) / 2
        w_offset = (x.shape[3] - self.crop_size) / 2
        if not isinstance(h_offset, torch.Tensor):
            h_offset, w_offset = torch.tensor(h_offset), torch.tensor(w_offset)
        h_end = h_offset + self.crop_size
        w_end = w_offset + self.crop_size
        return center_slice_helper(x, h_offset, w_offset, h_end, w_end)

class Normalize(nn.Module):
    def __init__(self, mean, std):
        """Subtract a constant mean from the tensor and divide by std
        :param mean: the constant mean to subtract
        :param std: the constant std to divide
        """
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if not x.dtype.is_floating_point:
            x = x.float()
        return (x - self.mean) / self.std

from torchvision.models import resnet34

net = resnet34(pretrained=True)
mean = torch.from_numpy(np.array([0.406, 0.456, 0.485], dtype=np.float32) * 255).view(-1, 1, 1)
std = torch.from_numpy(np.array([0.225, 0.224, 0.229], dtype=np.float32) * 255).view(-1, 1, 1)
dep_model = nn.Sequential(CenterCrop(224), Normalize(mean=mean, std=std), net, nn.Sigmoid())

onnxfile = "output/test.onnx"
dummy_input = torch.randn(1, 3, 341, 256, device='cpu').byte()
torch.onnx.export(dep_model, dummy_input, onnxfile, verbose=True, input_names=['data'], output_names=["prob"])

import torch
print("torch version:", torch.__version__)
import onnx
import onnxruntime

import torch.nn as nn
import numpy as np

class CenterCrop(nn.Module):
    def __init__(self, crop_size: int):
        """Crop from the center of a 4d tensor
        Input shape can be dynamic
        :param crop_size: the center crop size
        """
        super(CenterCrop, self).__init__()
        self.crop_size: int = crop_size

    def extra_repr(self):
        """Extra information
        """
        return 'crop_size={}'.format(
            self.crop_size
        )

    def forward(self, x):
        h_offset = (x.shape[2] - self.crop_size) // 2
        w_offset = (x.shape[3] - self.crop_size) // 2
        h_end = h_offset + self.crop_size
        w_end = w_offset + self.crop_size
        return x[:, :, h_offset:h_end, w_offset:w_end]

class Normalize(nn.Module):
    def __init__(self, mean, std):
        """Subtract a constant mean from the tensor and divide by std
        :param mean: the constant mean to subtract
        :param std: the constant std to divide
        """
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(x):
            x = x.float()
        return (x - self.mean) / self.std

from torchvision.models import resnet34

net = resnet34(pretrained=True)
mean = torch.from_numpy(np.array([0.406, 0.456, 0.485], dtype=np.float32) * 255).view(-1, 1, 1)
std = torch.from_numpy(np.array([0.225, 0.224, 0.229], dtype=np.float32) * 255).view(-1, 1, 1)
dep_model = torch.jit.script(nn.Sequential(CenterCrop(224), Normalize(mean=mean, std=std), net, nn.Sigmoid()))

onnxfile = "model.onnx"
dummy_input = torch.randn(1, 3, 341, 256, device='cpu').byte()
torch.onnx.export(
    dep_model, dummy_input, onnxfile, verbose=True, input_names=['data'],
    output_names=["prob"], opset_version=11)

onnx_model = onnx.load(onnxfile)
onnx.checker.check_model(onnx_model)
a = onnxruntime.InferenceSession(onnxfile)