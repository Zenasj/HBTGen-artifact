import torch
import torch.nn as nn
import torch.nn.functional as F

class Test(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.eye(512))
        self.conv = nn.Conv2d(3, 32, 3, 2, 1)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = torch.inverse(self.T).matmul(x)
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

if __name__ == '__main__':
    print(torch.__version__)

    model = Test().eval()

    x = torch.randn((1, 3, 512, 512))
    out = model(x)

    # TorchScript
    model_jit = torch.jit.trace(model, (x))
    model_jit.save("model_jit.pt")
    print(model_jit.code)

    # Optimize
    from torch.utils.mobile_optimizer import optimize_for_mobile
    model_jit_opt = optimize_for_mobile(model_jit)
    model_jit_opt.save("model_jit_opt.pth")
    print(model_jit_opt.code)

    model_jit_opt = model_jit_opt.eval()

    # export to Onnx
    torch.onnx.export(model_jit_opt,
        (x),
        'output.onnx',
        opset_version=11,
        verbose=True,
        example_outputs = out
    )

from typing import Dict, List
import torch
import torch.nn as nn
class FeatureNormalizer(nn.Module):
    def __init__(self, feature_names, is_map_negative : Dict[str, bool], types : Dict[str, str], params : Dict[str, float]):
        super(FeatureNormalizer, self).__init__()
        self.feature_names: List[str] = feature_names
        self.is_map_negative : Dict[str, bool] = is_map_negative
        self.types : Dict[str, str] = types
        self.params : Dict[str, float] = params
        self.means : Dict[str, float] = {}
        self.stds : Dict[str, float] = {}
        self.is_train: bool = False
    def forward(self, x):
        columns = []
        for i, feature_name in enumerate(self.feature_names):
            tmp = x[:, i].clone()
            if self.is_map_negative[feature_name]:
                tmp = torch.clamp(tmp, min=0.0)
            if self.types[feature_name] == 'log1p':
                tmp = torch.log1p(tmp)
            elif self.types[feature_name] == 'boxcox1p':
                lamb = self.params[feature_name]
                if lamb == 0.0:
                    tmp = torch.log1p(tmp)
                else:
                    tmp = (torch.pow((1.0 + tmp), lamb) - 1.0) / lamb
            
            if self.is_train:
                self.means[feature_name] = tmp.mean().item()
                self.stds[feature_name] = tmp.std().item()

            tmp = (tmp - self.means[feature_name]) / self.stds[feature_name]
            columns.append(tmp)
        res = torch.stack(columns, dim=1)
        return res

        
# Exporting:
inp = torch.zeros(1, 100).float().cpu()
out = torch.ones(1, 100).float().cpu()
scripted_module = torch.jit.script(norm_model.cpu())
torch.onnx.export(scripted_module, [inp], 
                  'normalizer.onnx', 
                  verbose=True, example_outputs=out,
                  opset_version=11)