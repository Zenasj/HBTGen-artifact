import torch
import torch.nn as nn
import torch.nn.functional as F

Output=self. sense (context_layer)

output=self. sense (context_layer),

class Linear4bit(nn.Linear):
    def __init__(self, input_features, output_features, bias=True, compute_dtype=None, compress_statistics=True, quant_type='fp4',device=None):
        super().__init__(input_features, output_features, bias, device)
        self.weight = Params4bit(self.weight.data, requires_grad=False, compress_statistics=compress_statistics, quant_type=quant_type)
        self.compute_dtype = compute_dtype
        self.compute_type_is_set = False

    def set_compute_type(self, x):
        if x.dtype in [torch.float32, torch.bfloat16]:
            # the input is in a dtype that is safe to compute in, we switch
            # to this type for speed and stability
            self.compute_dtype = x.dtype
        elif x.dtype == torch.float16:
            # we take the compoute dtype passed into the layer
            if self.compute_dtype == torch.float32 and (x.numel() == x.shape[-1]):
                # single batch inference with input torch.float16 and compute_dtype float32 -> slow inference when it could be fast
                # warn the user about this
                warnings.warn(f'Input type into Linear4bit is torch.float16, but bnb_4bit_compute_type=torch.float32 (default). This will lead to slow inference.')
                warnings.filterwarnings('ignore', message='.*inference.')
            if self.compute_dtype == torch.float32 and (x.numel() != x.shape[-1]):
                warnings.warn(f'Input type into Linear4bit is torch.float16, but bnb_4bit_compute_type=torch.float32 (default). This will lead to slow inference or training speed.')
                warnings.filterwarnings('ignore', message='.*inference or training')


    def forward(self, x: torch.Tensor):
        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        if getattr(self.weight, 'quant_state', None) is None:
            print('FP4 quantization state not initialized. Please call .cuda() or .to(device) on the LinearFP4 layer first.')
        if not self.compute_type_is_set:
            self.set_compute_type(x)
            self.compute_type_is_set = True

        inp_dtype = x.dtype
        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)

        bias = None if self.bias is None else self.bias.to(self.compute_dtype)
        out = bnb.matmul_4bit(x, self.weight.t(), bias=bias, quant_state=self.weight.quant_state) 

        out = out.to(inp_dtype)

        return out

def matmul_4bit(A: tensor, B: tensor, quant_state: List, out: tensor = None, bias=None):
    assert quant_state is not None
    if A.numel() == A.shape[-1] and A.requires_grad == False:
        absmax, shape, dtype, blocksize, compressed_stats, quant_type, data_type = quant_state
        if A.shape[-1] % blocksize != 0:
            warn(f'Some matrices hidden dimension is not a multiple of {blocksize} and efficient inference kernels are not supported for these (slow). Matrix input size found: {A.shape}')
            return MatMul4Bit.apply(A, B, out, bias, quant_state)
        else:
            out = F.gemv_4bit(A, B.t(), out, state=quant_state)
            if bias is not None:
                out += bias
            return out
    else:
        return MatMul4Bit.apply(A, B, out, bias, quant_state)