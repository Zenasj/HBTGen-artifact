import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
import torch.onnx

language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")

# Create dummy audio signal data 
signal = torch.zeros(48000)

prediction = language_id.classify_batch(signal)
print(prediction)

torch.onnx.export(language_id, signal, "langid.onnx", export_params=True, 
    do_constant_folding=True, input_names=['input'], output_names=['output'], 
    dynamic_axes={'input' : {0 : 'batch_size'}}, dynamo=True, report=True)

torch._subclasses.fake_tensor.UnsupportedOperatorException: aten._fft_r2c.default