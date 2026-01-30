import torch
import torch.nn as nn
import torch.nn.functional as F

def one_hot(labels, nc, device=None, dtype=None, eps=1e-6):
    batch_size, height, width, depth = labels.shape
    one_hot = torch.zeros(batch_size, nc, height, width, depth, device=device,
                          dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps

class TverskyLoss3d(nn.Module):
    def __init__(self, nc, tversky_beta=0.45, raw_logits=True):
        super().__init__()
        self.nc = nc
        self.raw_logits = raw_logits
        self.eps = 1e-6
        self.beta = tversky_beta
        self.alpha = 1 - self.beta

    def forward(self, pred, target):
        if self.raw_logits:
            pred_soft = F.softmax(pred, dim=1)
        else:
            pred_soft = pred.exp()
        ones = pred_soft*0 + 1.0

        target_one_hot = one_hot(target, nc=self.nc, device=pred.device,
                                 dtype=pred.dtype)

        dims = (0,2,3,4)
        intersection = (pred_soft * target_one_hot).sum(dims)
        fps = torch.sum(pred_soft * (ones - target_one_hot), dims)
        fns = torch.sum((ones - pred_soft) * target_one_hot, dims)

        numerator = intersection
        denominator = intersection + self.alpha * fps + self.beta * fns

        tversky_idx = \
        torch.ones((self.nc)).cuda(pred.device) - ((numerator + self.eps)/(denominator + self.eps))
        
        return tversky_idx.sum()

class ClampedConv3d(nn.Conv3d):
    '''
    Modified nn.Conv3d that clamps output to FP16 range avoiding infinity.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', **kwargs):
        self.fp16 = None
        if 'fp16' in kwargs:
            self.fp16 = kwargs['fp16']
            kwargs.pop('fp16')
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)

    def forward(self, input):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[2] + 1) // 2, self.padding[2] // 2,
                                (self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            output = F.conv3d(F.pad(input, expanded_padding, mode='circular'),
                              self.weight, self.bias, self.stride, expanded_padding,
                              self.dilation, self.groups)
            if self.fp16:
                output = torch.clamp(output, -6.55e4, 6.55e4)
            return output
        output = F.conv3d(input, self.weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
        if self.fp16:
            output = torch.clamp(output, -6.55e4, 6.55e4)
        return output

class seg3d_mod(nn.Module):
    def __init__(self, fp16=True):
        super().__init__()
        self.fp16 = fp16
        ...
    def forward(self, inp):
        with autocast(enabled=self.fp16):
            ...
            return x