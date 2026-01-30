import torch

from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
input_values = tokenizer(audio, return_tensors = "pt").input_values

model.eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_fp32_fused = torch.quantization.fuse_modules(model, [['conv', 'relu']],inplace=True)
model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)
model_fp32_prepared(input_values)
model_int8 = torch.quantization.convert(model_fp32_prepared)
res = model_int8(input_values)