import torch
import torch.nn as nn

def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            padded_input = torch.nn.functional.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            unfold_input = torch.nn.functional.unfold(padded_input,kernel_size)
            kernels_flat = self.weight.data.view(out_channels, -1)
            res = (kernels_flat @ unfold_input).view(1, out_channels, input.shape[2], input.shape[3])
            return res
        else:
            unfold_input = torch.nn.functional.unfold(input,kernel_size) #functional has unfold
            #kernels_flat = self.weight.data.view(out_channels, -1)
            res = (kernels_flat @ unfold_input).view(1, out_channels, input.shape[2], input.shape[3])
            return res

nn.functional.unfold()