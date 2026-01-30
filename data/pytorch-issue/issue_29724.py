import torch
import numpy as np
import random

def test_quantized_convert(self):
        torch.backends.quantized.engine = "qnnpack"
        module_quant = torch.jit.load("/home/pytorch/model_quantized.pt")
        x_numpy = np.random.random((2, 3, 8, 8)).astype("float32")
        X = torch.from_numpy(x_numpy)
        module_quant.eval()
        output = module_quant(X)
        torch.set_default_tensor_type(torch.FloatTensor)
        input_names = ["x"]
        f = io.BytesIO()
        torch.onnx.export(module_quant, (X), f, verbose=False, example_outputs=output, input_names=input_names, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=9)
        f.seek(0)
        onnx_model = onnx.load(f)
        sample_inputs = (
            x_numpy,
        )
        caffe_res = c2.run_model(onnx_model, dict(zip(input_names, sample_inputs)))[0]
        np.testing.assert_almost_equal(output.numpy(), caffe_res, decimal=3)