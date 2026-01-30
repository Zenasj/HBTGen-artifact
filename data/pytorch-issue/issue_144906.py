import torch

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# load model and processor
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

input_features = torch.randn(1,80, 3000)
attention_mask = torch.ones(1, 3000)
decoder_input_ids = torch.tensor([[1, 1, 1 , 1]]) * model.config.decoder_start_token_id

model.eval()

exported_program: torch.export.ExportedProgram= torch.export.export(model, args=(input_features, attention_mask, decoder_input_ids,), strict=True)