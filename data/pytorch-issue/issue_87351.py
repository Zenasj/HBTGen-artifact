import whisper, torch
model = whisper.load_model('small', device='mps')
whisper.decoding.DecodingTask(model, whisper.DecodingOptions()).run(torch.zeros([1, 80, 3000]).to(model.device))

import torch
torch.zeros(65).to('mps')[:] = 0