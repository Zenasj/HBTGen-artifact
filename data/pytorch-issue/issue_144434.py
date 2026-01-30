import torch.nn as nn

import torch
import torchvision
import tempfile
import time
import sys
import os

torch.manual_seed(42)

from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
# from torch.ao.quantization.quantizer.xnnpack_quantizer import (
#     XNNPACKQuantizer, get_symmetric_quantization_config,
# )

from torch.ao.quantization.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer, get_default_x86_inductor_quantization_config
)

import torch._inductor.config as config
config.cpp_wrapper = True

RUN_NUM = 1000

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p")/1e6)
    os.remove("temp.p")

def main(argv):
    workdir="/tmp"

    # Step 1: Capture the model
    args = (torch.randn(1, 3, 224, 224),)
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT).eval()
    m1 = m.eval()
    

    print("before size")
    print_size_of_model(m)


    with torch.no_grad():
        compiled = torch.compile(m1, backend="inductor")
        before = time.time()
        for i in range(RUN_NUM):
            compiled(*args)
        print(f"Python inference time: {(time.time() - before) / RUN_NUM}")


    # torch.export.export_for_training is only avaliable for torch 2.5+
    m = capture_pre_autograd_graph(m, args)

    # Step 2: Insert observers or fake quantize modules
    # quantizer = XNNPACKQuantizer().set_global(
    #     get_symmetric_quantization_config())
    quantizer = X86InductorQuantizer().set_global(
        get_default_x86_inductor_quantization_config())
    m = prepare_pt2e(m, quantizer)

    # Step 2.5 fake calibration
    m(*args)
    # # Step 3: Quantize the model
    m = convert_pt2e(m, fold_quantize=True)
    
    print(m)
    print("after size")
    print_size_of_model(m)


    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True), torch.no_grad():
        before_compile_time = time.time()
        compiled_quant = torch.compile(m, backend="inductor")
        print("finnished compiliation", time.time() - before_compile_time)
        before = time.time()
        for i in range(RUN_NUM // 100):
            compiled_quant(*args)
            # print("finished run", i)
        print(f"Python inference before quant time after quant: {(time.time() - before) / (RUN_NUM // 100)}")


    os.makedirs(workdir, exist_ok=True)

if __name__ == '__main__':
    main(sys.argv)

quantizer = X86InductorQuantizer().set_global(
                get_default_x86_inductor_quantization_config())
quantizer.set_module_type_qconfig(torch.nn.Linear, None)