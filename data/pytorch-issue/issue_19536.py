import torch
import torch.nn as nn
import math

def dropout_(tensor, p=0, mode='fan_in', nonlinearity='dropout_relu'):
    r"""Fills the input `Tensor` with values according to the method
    described in "Critical initialisation for deep signal propagation in
noisy rectifier neural networks" - Pretorius, A. et al. (2018). The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std})` where

    .. math::
        \text{std} = \sqrt{\frac{2 \times (1-p)}{\text{fan\_in}}}

    Args:
        tensor: an n-dimensional `torch.Tensor`
        p: probability of an element to be zeroed. This should be set equal to the dropout rate p in the subsequent dropout layer. 
        mode: either 'fan_in' (default) or 'fan_out'. Choosing `fan_in`
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing `fan_out` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with 'relu'.

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.dropout_(w, p=0.6, mode='fan_in', nonlinearity='relu')
    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, p)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std)