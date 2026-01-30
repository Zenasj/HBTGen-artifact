import torch

torch.onnx.export(
        torch_model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        "legacy_model_dynamic_axes_bsz_4.onnx",  
        opset_version=16,  # the ONNX version to export the model to
        verbose=True,
        do_constant_folding=True,
        input_names=["input_0", "input_1", "input_2", "input_3", "input_4", "input_5", "input_6", "input_7"],
        output_names=["output"],
        dynamic_axes={
            "input_0": {0:'batch'},
            "input_1": {0:'batch'},
            "input_2": {0:'batch'},
            "input_3": {0:'batch'},
            "input_4": {0:'batch'},
            "input_5": {0:'batch'},
            "input_6": {0:'batch'},
            "input_7": {0:'batch'},
        },
    )