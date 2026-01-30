import torch

def export_onnx(onnx_model, output, dynamic_axes, dummy_inputs, output_names):
    with open(output, "wb") as f:
        print(f"Exporting onnx model to {output}...")
        torch.onnx.export(
            onnx_model,
            tuple(dummy_inputs.values()),
            f,
            export_params=True,
            verbose=False,
            opset_version=17,
            do_constant_folding=True,
            input_names=list(dummy_inputs.keys()),
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )