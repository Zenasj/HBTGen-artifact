import torch
import torch.nn as nn

class MultiTaskVisionModelV4(nn.Module):
    def __init__(self, feature_extractor, feature_size, class_nums, is_quant=False):
        super(MultiTaskVisionModelV4, self).__init__()
        if is_quant:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
        self.is_quant = is_quant

        self.feature_extractor = feature_extractor
        self.pooling = nn.AdaptiveMaxPool2d(1)
        self.fc_object = nn.Linear(feature_size, class_nums[0])

    def forward(self, x):
        if self.is_quant:
            x = self.quant(x)

        feats = self.feature_extractor(x)
        feat = feats[-1]
        pool_op = self.pooling(feat)  # object and place

        out_op = pool_op.flatten(1)

        o1 = self.fc_object(out_op)
        if self.is_quant:
            o1 = self.dequant(o1)

        return o1


def fuse_mobilenetv2(model):
    if hasattr(model, "module"):
        model = model.module

    fuse_list = ['feature_extractor.conv_stem', 'feature_extractor.bn1', 'feature_extractor.act1']
    torch.quantization.fuse_modules(model, fuse_list, inplace=True)

    for name, module in model.feature_extractor.named_modules():
        if isinstance(module, DepthwiseSeparableConv):
            fuse_list = [['conv_dw', 'bn1', 'act1'],
                         ['conv_pw', 'bn2'],
                         ]
            torch.quantization.fuse_modules(module, fuse_list, inplace=True)
        elif isinstance(module, InvertedResidual):
            fuse_list = [['conv_pw', 'bn1', 'act1'],
                         ['conv_dw', 'bn2', 'act2'],
                         ['conv_pwl', 'bn3'],
                         ]
            torch.quantization.fuse_modules(module, fuse_list, inplace=True)
            

def convert2qat_model(model_fp32):
    model_fp32.train()
    model_fp32.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    torch.quantization.prepare_qat(model_fp32, inplace=True)
    return model_fp32


# modified mobilenetv2_120d
backbone = timm.create_model("mobilenetv2_120d", features_only=True, pretrained=False, out_indices=(1, 2, 3, 4))
model = MultiTaskVisionModelV4(backbone, feature_size=384, class_nums=520, is_quant=True)
fuse_mobilenetv2(model)
convert2qat_model(model)

backbone = timm.create_model("mobilenetv2_120d", features_only=True, pretrained=False, out_indices=(1, 2, 3, 4))
model = MultiTaskVisionModelV4(backbone, feature_size=384, class_nums=520, is_quant=True)
model.eval()
checkpoint = torch.load("../checkpoints/pretrained_mbv2.pth", map_location="cpu")
sd = checkpoint["state_dict"]
if next(iter(sd.items()))[0].startswith('module'):
    sd = {k[len('module.'):]: v for k, v in sd.items()}
model.load_state_dict(sd)

model.train()
fuse_mobilenetv2(model)
model.eval()

model.train()
fuse_mobilenetv2(model)
convert2qat_model(model)
model.eval()

def _forward(self, input):
        assert isinstance(self.bn.running_var, torch.Tensor)
        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1

        scaled_weight = self.weight_fake_quant(self.weight * scale_factor.reshape(weight_shape))
        # using zero bias here since the bias for original conv
        # will be added later
        if self.bias is not None:
            zero_bias = torch.zeros_like(self.bias)
        else:
            zero_bias = torch.zeros(self.out_channels, device=scaled_weight.device)
        conv = self._conv_forward(input, scaled_weight, zero_bias)
        conv_orig = conv / scale_factor.reshape(bias_shape)
        if self.bias is not None:
            conv_orig = conv_orig + self.bias.reshape(bias_shape)
        conv = self.bn(conv_orig)
        return conv