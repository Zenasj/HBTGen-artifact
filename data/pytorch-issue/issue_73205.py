from typing import Tuple

import torch
from torch import tensor, randperm, Tensor
from torch.nn import CrossEntropyLoss


def label_smoothing_bug():
    with torch.no_grad():
        log_probs, target = _create_inputs()
        log_probs_masked = log_probs[target != -100]
        target_masked = target[target != -100]

        for label_smoothing in [0.0, 0.2]:
            for reduction in ["sum", "mean"]:
                loss_obj = CrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing)
                standard_loss = loss_obj(log_probs, target)
                masked_loss = loss_obj(log_probs_masked, target_masked)
                mimic_loss = mimic_incorrect_pytorch_smooth_loss(
                    log_probs, log_probs_masked, target_masked, reduction, label_smoothing)
                print(f"label_smoothing = {label_smoothing}, reduction = {reduction}, "
                      f"is correct? {torch.allclose(standard_loss, masked_loss)}, "
                      f"behavior mimicked? {torch.allclose(standard_loss, mimic_loss)}")


def mimic_incorrect_pytorch_smooth_loss(log_probs, log_probs_masked, target_masked, reduction, label_smoothing):
    n_classes = log_probs.shape[-1]
    uniform_target = torch.ones_like(log_probs) / n_classes
    loss_obj = CrossEntropyLoss(reduction=reduction)
    one_hot_loss = loss_obj(log_probs_masked, target_masked)
    uniform_loss = loss_obj(log_probs, uniform_target)  # includes ignored logprobs!
    incorrect_smooth_loss = (1.0 - label_smoothing) * one_hot_loss + label_smoothing * uniform_loss
    return incorrect_smooth_loss


def _create_inputs() -> Tuple[Tensor, Tensor]:
    bsz = 100
    classes = 35
    ignore_index = -100
    num_ignore = 30

    target = torch.randint(high=classes, size=[bsz])
    ignore_mask = tensor([True] * num_ignore + [False] * (bsz - num_ignore))
    ignore_mask = ignore_mask[randperm(bsz)]
    target[ignore_mask] = ignore_index

    logits = torch.randn(bsz, classes)
    log_probs = logits.log_softmax(dim=-1)

    return log_probs, target


if __name__ == '__main__':
    label_smoothing_bug()

from torch import Tensor
from torch.nn import CrossEntropyLoss


class FixedCrossEntropyLoss(CrossEntropyLoss):
    """
    Standard CrossEntropyLoss with label_smoothing doesn't handle ignore_index properly, so we apply
    the mask ourselves. See https://github.com/pytorch/pytorch/issues/73205
    """

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if not target.is_floating_point() and self.ignore_index is not None:
            input = input[target != self.ignore_index]
            target = target[target != self.ignore_index]
        loss = super().forward(input, target)
        return loss