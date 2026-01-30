import torch
import torch.nn as nn
import torch.nn.functional as F

var = x.var()
x = x - x.mean()
x = x/var

class BiGroupNorm(torch.nn.GroupNorm):
    """
    this is a binary implementation of torch.nn.GroupNorm
    """
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True, device=None,
                 dtype=None) -> None:
        super(BiGroupNorm, self).__init__(num_groups, num_channels, eps, affine, device, dtype)
        self._logger = logging.getLogger()

    def forward(self, input_: torch.Tensor) -> torch.Tensor:

        self.weight = Parameter(self.weight - self.weight.mean())
        self.weight = Parameter(BinaryQuantize().apply(self.weight))
        self.bias = Parameter(BinaryQuantize().apply(self.bias))
       
        # NOTICE
        # the following two lines suggest weight and bias is always zeros
        self._logger.debug('weight shape of {}: {}, sum of abs weight: {}'.format(self._get_name(), self.weight.shape,
                                                                              torch.sum(torch.abs(self.weight))))
        self._logger.debug('bias shape of {}: {}, sum of abs bias: {}'.format(self._get_name(), self.bias.shape,
                                                                              torch.sum(torch.abs(self.bias))))

        return F.group_norm(
            input_, self.num_groups, self.weight, self.bias, self.eps)

class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input_):
        """
        input_[ input_ >0 ] = 1
        input_[ input_ <0 ] = -1
        """
        ctx.save_for_backward(input_)
        out = torch.sign(input_)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_ = ctx.saved_tensors
        grad_input = grad_output
        grad_input[input_[0].gt(1)] = 0 
        grad_input[input_[0].lt(-1)] = 0 
        return grad_input