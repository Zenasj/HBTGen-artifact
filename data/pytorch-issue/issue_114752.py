import torch
import torch.nn as nn
import torch.nn.functional as F
aten = torch.ops.aten

@torch.compile
def linear_interpolation(features, input_fps, output_fps, output_len):
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    output_len = int(seq_len * output_fps + 1)
    output_features = aten.upsample_linear1d(features,[output_len],align_corners=True)
    return output_features.transpose(1, 2)

sizesTen = [[1, 336, 512], [1, 180, 512], [1, 233, 512], 
[1, 300, 512], [1, 278, 512], [1, 221, 512], [1, 256, 512]]
frameLen = [202, 109, 140, 181, 167, 133, 154]

for i,j in zip(sizesTen, frameLen):
    feat = torch.randn(*i).to(torch.float32).cuda()
    output = linear_interpolation(feat, 50, 30, j)