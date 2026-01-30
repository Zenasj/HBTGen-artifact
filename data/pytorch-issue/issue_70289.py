import torch.nn as nn

import torch
import torchaudio
import numpy as np
import soundfile as sf
import torch.nn.functional as F

import onnx
import onnxruntime

device="cpu"
# https://pytorch.org/audio/stable/pipelines.html#hubert-large
bundle = torchaudio.pipelines.HUBERT_LARGE
model = bundle.get_model().to(device)

audio_file = "sample.wav"   # shape: torch.Size([1, 101467])
x, sr = sf.read(audio_file, dtype='float32')
x = torch.Tensor(x).unsqueeze(0).cpu()
x = F.layer_norm(x, x.shape)

model_path = "torchaudio_hubert_large.onnx"
torch.onnx.export(model, x, 'torchaudio_hubert_large.onnx', input_names=['input'], output_names=['output'])
model = onnx.load(model_path)
model.graph.input[0].type.tensor_type.shape.dim[1].dim_param = '?'
onnx.save(model, model_path.replace(".onnx", "_dyn.onnx"))

model_path = "torchaudio_hubert_large_dyn.onnx"
ort_session = onnxruntime.InferenceSession(model_path)
feat = ort_session.run(None, {'input': x.numpy().astype(np.float32)})

import torch
import torchaudio
import numpy as np
import soundfile as sf
import torch.nn.functional as F

import onnx
import onnxruntime

device = "cpu"
# https://pytorch.org/audio/stable/pipelines.html#hubert-large
bundle = torchaudio.pipelines.HUBERT_LARGE
model = bundle.get_model().to(device)

# audio_file = "sample.wav"  # shape: torch.Size([1, 101467])
# x, sr = sf.read(audio_file, dtype="float32")
# x = torch.Tensor(x).unsqueeze(0).cpu()
# x = F.layer_norm(x, x.shape)
x = torch.randn(1, 101467)

model_path = "torchaudio_hubert_large.onnx"
torch.onnx.export(
    model,
    x,
    "torchaudio_hubert_large.onnx",
    input_names=["input"],
    output_names=["output"],
)

model = onnx.load(model_path)
model.graph.input[0].type.tensor_type.shape.dim[1].dim_param = '?'
onnx.save(model, model_path.replace(".onnx", "_dyn.onnx"))

model_path = "torchaudio_hubert_large_dyn.onnx"
ort_session = onnxruntime.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
feat = ort_session.run(None, {'input': x.numpy().astype(np.float32)})
print(feat)