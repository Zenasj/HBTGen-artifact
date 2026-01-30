import torch

from torch.utils.cpp_extension import load_inline
at_ops = load_inline(
        name='at_ops',
        cpp_sources=["""
        using namespace at;
        """],
        functions=[
            "mkldnn_convolution",
            "mkldnn_convolution_backward",
            #"cudnn_convolution",
            "cudnn_convolution_backward",
            "cudnn_grid_sampler",
            "cudnn_grid_sampler_backward"
        ],
        with_cuda=True
)

from torch.utils.cpp_extension import load_inline
at_ops = load_inline(
        name='at_ops',
        cpp_sources=["""
        using namespace at;
        """],
        functions=[
            "mkldnn_convolution",
            "mkldnn_convolution_backward",
            "cudnn_convolution",
            "cudnn_convolution_backward",
            "cudnn_grid_sampler",
            "cudnn_grid_sampler_backward"
        ],
        with_cuda=True
)

from torch.utils.cpp_extension import load_inline
at_ops = load_inline(
        name='at_ops',
        cpp_sources=["""
        using namespace at;
        """],
        functions=[
            "cudnn_convolution",
        ],
        with_cuda=True
)