import torch
import torchaudio
import torch.onnx

jit_preprocessor = torch.jit.load("tmp.pt")

wav_file = "/home/divyansh/AAT3_16khz.wav"
signal, sample_rate = torchaudio.load(wav_file)
length = torch.tensor(signal.size(1)).unsqueeze(0)

input_names = ["input_signal", "length"]
output_names = ["output_features", "output_lengths"]
onnx_file = "NeMoPreprocessor.onnx"

jit_preprocessor.eval()

torch.onnx.dynamo_export(
    jit_preprocessor,
    (signal, length),
    onnx_file,
    verbose=True,
    opset_version=18
)

tmp.pt