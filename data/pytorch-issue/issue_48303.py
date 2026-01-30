import torch
import torch.nn as nn
import torch.nn.functional as F

torch.nn.functional.cosine_similarity

class exampleNet(nn.Module):
    def __init__(self):
        super(exampleNet, self).__init__()

    def forward(self, x1, x2):
        return F.cosine_similarity(x1, x2, dim=-1)

n = exampleNet().eval()
x1 = torch.rand((1, 32, 32, 256))
x2 = torch.rand((1, 32, 32, 256))
r = n(x1,x2)
torch.onnx.export(n, (x1, x2,), "model.onnx", export_params=True, opset_version=12)

def cosine_similarity_onnx_exportable(x1, x2, dim=-1):
    cross = (x1 * x2).sum(dim=dim)
    x1_l2 = (x1 * x1).sum(dim=dim)
    x2_l2 = (x2 * x2).sum(dim=dim)
    return torch.div(cross, (x1_l2 * x2_l2).sqrt())