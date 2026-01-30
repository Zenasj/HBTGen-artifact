import torch.nn as nn

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
input_values = tokenizer(audio, return_tensors = "pt").input_values
model_int8 = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear}, 
    dtype=torch.qint8)