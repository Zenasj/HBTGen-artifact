#pip install git+https://github.com/mobiusml/gemlite/ -> Need to comment out the triton.autotune extra params like prune_configs_by since torch.compile doesn't support that yet
###############################################################
from gemlite.core import GemLiteLinearTriton, DType
import torch 

def check_valid(x, W, quant_linear, tol=1e-3):
    y_ref = torch.matmul(x, W.T)
    y_q   = quant_linear(x)
    try:
        assert (y_ref - y_q).abs().mean() < tol
    except:
        raise Exception('Assertion Failed')

W_nbits, group_size = 4, 128 
in_features, out_features = 4096, 4096

gemlite_linear = GemLiteLinearTriton(W_nbits, group_size=group_size, 
                                    in_features=in_features, 
                                    out_features=out_features, 
                                    input_dtype=DType.FP16, 
                                    output_dtype=DType.FP16, 
                                    acc_dtype=DType.FP16)



###############################################################
device = 'cuda:0'
compute_dtype = torch.float16

orig_shape = (out_features, in_features)

W_q    = torch.randint(0, 2**W_nbits, (out_features, in_features), dtype=torch.uint8, device=device).to(torch.uint8)
N      = in_features * out_features // group_size
scales = torch.randn((N,), dtype=compute_dtype, device=device).abs()/500.
zeros  = torch.randint(0, 2**W_nbits - 1, (N,), dtype=compute_dtype, device=device)
W      = ((W_q.reshape([-1, group_size]) - zeros.view((N, 1))) * scales.view((N, 1))).reshape(orig_shape)

gemlite_linear.pack(W_q, scales, zeros, None);
###############################################################

for batch_size in [16]:
    x = torch.randn((batch_size, in_features), dtype=gemlite_linear.compute_dtype, device='cuda:0')/10.
    check_valid(x, W, gemlite_linear)
#OK

gemlite_linear.forward = torch.compile(gemlite_linear.forward, fullgraph=True)

for batch_size in [16]:
    x = torch.randn((batch_size, in_features), dtype=gemlite_linear.compute_dtype, device='cuda:0')/10.
    check_valid(x, W, gemlite_linear)

#Incorrect