import torch
import torch.nn as nn
import numpy as np

def compare_model_outputs_onnx(model: nn.Module, model_onnx: ort.InferenceSession, data_generator: torch.utils.data.DataLoader[Tuple[torch.Tensor, torch.Tensor, str]]) -> None:
    input_name = model_onnx.get_inputs()[0].name
    output_diffs = []
    for image_tensor, _, _ in itertools.islice(data_generator, 10000):
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()

        # compute ONNX Runtime output prediction
        onnx_input = to_numpy(image_tensor)

        ort_inputs = {input_name: onnx_input}
        ort_outs = model_onnx.run(None, ort_inputs)

        torch_output = to_numpy(model(image_tensor))
        onnx_output = ort_outs[0]

        # compare ONNX Runtime and PyTorch results

        if not np.allclose(torch_output, onnx_output, rtol=1e-03, atol=1e-5):
            print(f"The max difference {np.abs(onnx_output - torch_output).max()} is larger than the atol {5e-5}")
            raise RuntimeError("Prediction of onnx model model not identical to trained model")