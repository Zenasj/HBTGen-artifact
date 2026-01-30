import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm

def subnet_conv(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, 256,   3, padding=1), nn.ReLU(), nn.Conv2d(256,  c_out, 3, padding=1))

def subnet_conv_1x1(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, 256,   1), nn.ReLU(), nn.Conv2d(256,  c_out, 1))

decoder = Ff.SequenceINN(256, 54, 54) # Init with dimension w/o batch size
for k in range(2):
    decoder.append(Fm.GLOWCouplingBlock, subnet_constructor=subnet_conv)
    decoder.append(Fm.GLOWCouplingBlock, subnet_constructor=subnet_conv_1x1)

traced_decoder = torch.jit.trace(decoder, torch.empty(1, 256, 54, 54))
trt_model = torch_tensorrt.compile(traced_decoder, inputs = [torch_tensorrt.Input((1, 256, 54, 54), dtype=torch.float32)],
                                   enabled_precisions = torch.float32)