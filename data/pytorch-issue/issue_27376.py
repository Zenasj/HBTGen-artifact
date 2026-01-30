import torch.nn.functional as F

if self.training:
    x = F.interpolate(x, scale_factor=2, mode="nearest")
else:
    # hack in order to generate a simpler onnx
    x = F.interpolate(x, size=[int(2 * x.shape[2]), int(2 * x.shape[3])], mode='nearest')