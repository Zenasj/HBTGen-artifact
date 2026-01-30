import torch.nn as nn

python
import torch
import pickle


assert torch.cuda.is_available() and torch.backends.cudnn.enabled

# https://drive.google.com/file/d/1eltV2WjDVTuRBXO0n6E5t5dkQ3GEW6Qj/view?usp=sharing
with open('ctc_bug.pkl', 'rb') as f:
    variables = pickle.load(f)

log_probs = variables['log_probs'].cuda()
targets = variables['targets'].cpu()
target_lengths = variables['target_lengths'].cuda()

# with `zero_infinity=False` it works well
ctc_loss = torch.nn.CTCLoss(zero_infinity=True).cuda()

assert torch.isfinite(log_probs.max()) and torch.isfinite(log_probs.min())

# with `torch.backends.cudnn.enabled = False` it works well
T, N, C = log_probs.shape
loss = ctc_loss(
    log_probs=log_probs,
    targets=targets,
    input_lengths=torch.full(size=(N,), fill_value=T, dtype=torch.int32).cuda(),
    target_lengths=target_lengths
)
assert torch.isfinite(loss)
loss.backward()