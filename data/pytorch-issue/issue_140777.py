import torch
import torch.nn as nn


ctc = nn.CTCLoss(blank=0, reduction='none', zero_infinity=True)

logp, targets, input_lens, target_lens = torch.load("ctc-sanitizer.pt")
logp.requires_grad_(True)

logp, targets, input_lens, target_lens = (
    logp.to(device="cuda:0"), targets.to(device="cuda:0"),
    input_lens.to(device="cuda:0"), target_lens.to(device="cuda:0"),
)
loss = ctc(logp, targets, input_lens, target_lens).sum()
loss.backward()

def check_ctc_inputs(log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor):
    assert log_probs.dim() == 3, "log_probs should be a 3D tensor"
    assert targets.dim() == 2, "targets should be a 2D tensor"
    assert input_lengths.dim() == 1, "input_lengths should be a 1D tensor"
    assert target_lengths.dim() == 1, "target_lengths should be a 1D tensor"

    T, N, C = log_probs.shape
    assert targets.shape[0] == N, "Batch size mismatch"
    assert input_lengths.shape[0] == N, "Batch size mismatch"
    assert target_lengths.shape[0] == N, "Batch size mismatch"

    assert torch.all(input_lengths <= T), "input_lengths should be less than or equal to T"
    assert torch.all(input_lengths >= target_lengths), "input lengths must be >= target lengths"

    assert torch.all(targets < log_probs.shape[-1]), "targets should be smaller than number of classes"
    assert torch.all(target_lengths <= targets.shape[1]), "target lengths longer than targets tensor"

    assert torch.all(targets >= 0), "targets (including padding) should be >= 0"
    for i in range(N):
        t = targets[i, :target_lengths[i]]
        assert torch.all(t > 0), "targets (excluding padding) should be positive integers"

        nonrepeat = torch.unique_consecutive(t, dim=-1)
        assert torch.all(t == nonrepeat), "targets contains consecutively duplicated tokens"

    assert torch.all(log_probs.isfinite()), "log probs has infs or nans"

logp = logp[:, :1024]
targets = targets[:1024]
input_lens = input_lens[:1024]
target_lens = target_lens[:1024]