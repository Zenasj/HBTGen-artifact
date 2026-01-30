import torch

def export_quantize(m):
    # Issue: bn expected 4d, got 3
    from torch._export import capture_pre_autograd_graph
    example_inputs = (torch.randn(1, 3, 224, 224),) # Note: input should be a tuple
    # breakpoint()
    m = capture_pre_autograd_graph(m, example_inputs)
    # we get a model with aten ops


    # Step 2. quantization
    from torch.ao.quantization.quantize_pt2e import (
    prepare_pt2e,
    convert_pt2e,
    )

    from torch.ao.quantization.quantizer.xnnpack_quantizer import ( # Note: Updated import path
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
    # backend developer will write their own Quantizer and expose methods to allow
    # users to express how they
    # want the model to be quantized
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
    m = prepare_pt2e(m, quantizer)

    # calibration omitted

    m = convert_pt2e(m)
    # we have a model with aten ops doing integer computations when possible

    return m