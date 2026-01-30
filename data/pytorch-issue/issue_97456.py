import torch

model_int8 = torch.quantization.quantize_dynamic(model, None, dtype=torch.qint8)
mic_audio = torch.randn((4, frame_number))* 32367
echo_ref_audio = torch.rand((1, frame_number))* 32367
cache_new = torch.zeros_like(cache) + 1.0e-8
export_inputs = (mic_audio, echo_ref_audio, cache_new)
torch.onnx.export(model_int8, export_inputs, output_filename, verbose=True, opset_version=16)