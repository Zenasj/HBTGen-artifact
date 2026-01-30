{python}
import whisper
import torch
device = torch.device('mps')
model.transcribe("test dictation.m4a",language='english',verbose=True,fp16=False)