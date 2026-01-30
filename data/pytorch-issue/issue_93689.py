import torch
import torch.nn as nn
import torchdynamo

tensor_dtype = torch.float16

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x, y):
        output_size = (x.size(0), x.size(1), y.size(2))     

        matmul_result = torch.empty(
            output_size[0],
            output_size[1],
            output_size[2],
            dtype=x.dtype,
            device=torch.cuda.current_device())

        out = torch.baddbmm(matmul_result,
                                      x,
                                      y,
                                      beta=0, alpha=0.1)

        
        return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Net().cuda().half()

x = torch.rand((16, 1024, 16), dtype=tensor_dtype, requires_grad=True, device=device)
y = torch.rand((16, 16, 1024), dtype=tensor_dtype, requires_grad=True, device=device)

network_fn = torchdynamo.optimize("aot_nvfuser")(net)
#network_fn = torch.jit.trace(net, (x,y))

outputs = network_fn(x, y)