import torch
import torch.nn as nn

class UC( nn.Module):
    def forward( self, x):
        for _ in range( 8):
            x = x[:,:,::2]
        for _ in range( 8):
            x = x.repeat_interleave( 2, -1)
        return x

torch.onnx.export( UC(), torch.randn( 1, 128, 1024), 'foo.pth', opset_version=14)

[1, 128, 4] [1, 128, 4]
[0, 0, 0] [-1, -1, -1]
[1, 128, 16] [1, 128, 16]
[0, 0, 0] [-1, -1, -1]
[1, 128, 64] [1, 128, 64]
[0, 0, 0] [-1, -1, -1]
[1, 128, 256] [1, 128, 256]
[0, 0, 0] [-1, -1, -1]