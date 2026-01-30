import torch

from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

model = AutoModelForAudioClassification.from_pretrained("alefiury/wav2vec2-xls-r-300m-pt-br-spontaneous-speech-emotion-recognition")

model.eval()

input_values = torch.rand((1,8000))
attention_mask = torch.ones(1,8000)

exported_program: torch.export.ExportedProgram= torch.export.export(model, args=(input_values,attention_mask,), strict=False)