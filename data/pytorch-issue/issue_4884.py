import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_cuda_kernel_size_check():
    # RuntimeError(Expected tensor for argument #1 'input' to have the same dimension
    # as tensor for 'result'; but 4 does not equal 2 (while checking arguments for cudnn_convolution))
    conv = nn.Conv2d(1, 1, kernel_size=4, padding=1).cuda()
    x = Variable(torch.randn(1, 1, 1, 5).cuda())
    return conv(x)

def conv2d_thcunn_kernel_size_check_with_stride():
    # 2d case is expected
    # RuntimeError(Calculated input size: (3 x 7). Kernel size: (4 x 4). Kernel size
    # can't greater than actual input size at /home/ssnl/sftp/pytorch/aten/src/THCUNN/generic/SpatialConvolutionMM.cu:48)
    try:
        torch.backends.cudnn.enabled = False
        conv = nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=1).cuda()
        x = Variable(torch.randn(1, 1, 1, 5).cuda())
        return conv(x)
    finally:
        torch.backends.cudnn.enabled = True

def conv3d_thcunn_kernel_size_check_with_stride():
    # 3d case has no error (unexpected)
    try:
        torch.backends.cudnn.enabled = False
        conv = nn.Conv3d(1, 1, kernel_size=4, stride=2, padding=1).cuda()
        x = Variable(torch.randn(1, 1, 1, 5, 5).cuda())
        return conv(x)
    finally:
        torch.backends.cudnn.enabled = True

def conv_cudnn_kernel_size_check_with_stride():
    # both 2d and 3d give CUDNN_STATUS_BAD_PARAM
    conv = nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=1).cuda()
    x = Variable(torch.randn(1, 1, 1, 5).cuda())
    return conv(x)

def batch_norm_missing_var():
    # in 0.3 it thorws RuntimeError(_Map_base::at)
    # it is better now RuntimeError(Variable data has to be a tensor, but got Variable)
    # but still somewhat confusing.
    F.batch_norm(
        Variable(torch.randn(1, 2, 2)),   # input
        torch.zeros(2),            # running mean
        #torch.zeros(2),          # running var
        Variable(torch.ones(2)),  # gain
        Variable(torch.zeros(2)),  # bias
        True,                      # is_training
        0.1,                       # 1 - momentum
        1e-5,                      # epsilon
    )