import torch

classifier = EncoderClassifier.from_hparams(source="best_model/", hparams_file='hparams_inference.yaml', savedir="best_model/")
audio_file = 'test_output.wav'
signal, fs = torchaudio.load(audio_file)

with torch.no_grad():
    torch.onnx.export(
        classifier,
        signal,
        "mymodel.onnx",
        opset_version=11,
        input_names=['input'],
        output_names=['output'])