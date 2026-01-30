import torch
import torchvision
import copy
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch._dynamo.backends.common import aot_autograd


def my_compiler(gm, example_inputs):
    print(gm)
    return gm

def pt2e_ptq(m, example_inputs):
    m = m.eval()
    exported_model = torch.export.export_for_training(m, example_inputs).module()
    quantizer = X86InductorQuantizer()
    quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())

    prepared_model = prepare_pt2e(exported_model, quantizer)
    _ = prepared_model(*example_inputs)
    converted_model = convert_pt2e(prepared_model)
    torch.ao.quantization.move_exported_model_to_eval(converted_model)

    with torch.no_grad():
        my_backend = aot_autograd(fw_compiler=my_compiler)
        optimized_model = my_backend(converted_model, example_inputs) # !!! error here !!!

    optimized_model(*example_inputs)


if __name__ == "__main__":
    data = torch.randn(16, 3, 224, 224)
    model_fp = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    pt2e_ptq(copy.deepcopy(model_fp), (data,))

import torch
import torchvision
import copy
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import make_boxed_func
from torch._inductor.fx_passes.freezing_patterns import freezing_passes


def my_compiler(gm, example_inputs):
    freezing_passes(gm, example_inputs) # !!! freezing passes are applied but constant_fold passes are not !!!
    print(gm)
    return make_boxed_func(gm.forward)

my_backend = aot_autograd(fw_compiler=my_compiler)


def pt2e_ptq(m, example_inputs):
    m = m.eval()
    exported_model = torch.export.export_for_training(m, example_inputs).module()
    quantizer = X86InductorQuantizer()
    quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
    prepared_model = prepare_pt2e(exported_model, quantizer)
    _ = prepared_model(*example_inputs)
    converted_model = convert_pt2e(prepared_model)
    torch.ao.quantization.move_exported_model_to_eval(converted_model)

    with torch.no_grad():
        optimized_model = torch.compile(converted_model, backend=my_backend)

    _ = optimized_model(*example_inputs)

if __name__ == "__main__":

    data = torch.randn(16, 3, 224, 224)
    model_fp = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    pt2e_ptq(copy.deepcopy(model_fp), (data,))

import torch
import torchvision
import copy
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch._dynamo.backends.common import aot_autograd
from torch._inductor.fx_passes.freezing_patterns import freezing_passes
from torch._inductor.constant_folding import constant_fold


def _post_autograd_decomp_table():
    decomp_table = torch.export.default_decompositions()

    # if we are post-autograd, we shouldn't
    # decomp prim ops.
    for k in list(decomp_table.keys()):
        if not torch._export.utils._is_cia_op(k):
            del decomp_table[k]

    return decomp_table

def pt2e_ptq(m, example_inputs):
    m = m.eval()
    exported_model = torch.export.export_for_training(m, example_inputs).module()
    quantizer = X86InductorQuantizer()
    quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())

    prepared_model = prepare_pt2e(exported_model, quantizer)
    _ = prepared_model(*example_inputs)
    converted_model = convert_pt2e(prepared_model)
    torch.ao.quantization.move_exported_model_to_eval(converted_model)

    with torch.no_grad():
        optimized_model = torch.export.export_for_training(
            converted_model, example_inputs
        ).run_decompositions(_post_autograd_decomp_table()).module()
        freezing_passes(optimized_model, example_inputs)
        constant_fold(optimized_model)
        print(optimized_model)

    optimized_model(*example_inputs)


if __name__ == "__main__":
    data = torch.randn(16, 3, 224, 224)
    model_fp = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    pt2e_ptq(copy.deepcopy(model_fp), (data,))