import torch
import torch.nn as nn

SWAPPABLE_MODULES = (nni.ConvBn2d,
                     nni.ConvBnReLU2d,
                     nni.LinearReLU,
                     nni.ConvReLU2d)

def quantize_shadow(model, run_fn, run_args, mapping=None, inplace=False):
    if mapping is None:
        mapping = DEFAULT_MODULE_MAPPING
    if not inplace:
        model = copy.deepcopy(model)
    model.eval()
    prepare(model, inplace=True)
    run_fn(model, run_args)
    transform_shadow(model, mapping, inplace=True)
    return model

def transform_shadow(module, mapping=None, inplace=False):
    if mapping is None:
        mapping = DEFAULT_MODULE_MAPPING
    if not inplace:
        module = copy.deepcopy(module)
    reassign = {}
    # TODO(jerryzh): remove after deciding on the impl of intrinsic modules
    # This is required because intrinsic modules right now are implemented as
    # nn.Sequential and we don't want to swap their constituents
    SWAPPABLE_MODULES = (nni.ConvBn2d,
                         nni.ConvBnReLU2d,
                         nni.LinearReLU,
                         nni.ConvReLU2d)

    for name, mod in module.named_children():
        if type(mod) not in SWAPPABLE_MODULES:
            transform_shadow(mod, mapping, inplace=True)
        if type(mod) in SWAPPABLE_MODULES:
            reassign[name] = add_shadow_module(mod, mapping)
        else:
            reassign[name] = swap_module(mod, mapping)

    for key, value in reassign.items():
        module._modules[key] = value

    return module

def add_shadow_module(mod, mapping):
    new_mod = mod
     # Always replace dequantstub with dequantize
    if hasattr(mod, 'qconfig') and mod.qconfig is not None or type(mod) == DeQuantStub:
        if type(mod) in mapping:
            if type(mod) == QuantStub or type(mod) == DeQuantStub:
                new_mod = mapping[type(mod)].from_float(mod)
            else:
                qmod = mapping[type(mod)].from_float(mod)
                new_mod = Shadow(qmod, mod)
    return new_mod

class Shadow(nn.Module):
    def __init__(self, q_module, float_module):
        super(Shadow, self).__init__()
        self.orig_module = q_module
        self.shadow_module = float_module
        self.dequant = nnq.DeQuantize()

        self.orig_ob = torch.quantization.RecordingObserver()
        self.shadow_ob = torch.quantization.RecordingObserver()

    def forward(self, x):
        output = self.orig_module(x)

        # Output is in float for dynamic quantization
        if output.is_quantized:
            self.orig_ob(output.dequantize())
        else:
            self.orig_ob(output)

        # Handle dynamic quantization
        if x.is_quantized:
            x = x.dequantize()

        shadow_output = self.shadow_module(x)
        self.shadow_ob(shadow_output)
        return output

ob_dict = get_observer_dict(model)