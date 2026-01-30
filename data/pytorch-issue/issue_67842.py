import torch
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)
model._save_for_lite_interpreter("AndroidStudioProjects/testVAD/app/src/main/assets/silero_vad.ptl")