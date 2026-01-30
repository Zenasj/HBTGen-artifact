import torch

torch.autograd.gradcheck(lambda x: ctc_loss(x.log_softmax(2), labels, seqs, label_sizes), gpu_log_probs.double(), atol=1e-4)