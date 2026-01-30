import torch
from torch import Tensor


@torch.jit.script
def multilabel_confusion_matrix(y_true: Tensor, y_pred: Tensor) -> Tensor:
    true_and_pred = torch.mul(y_true, y_pred)
    tp_sum = torch.count_nonzero(true_and_pred, dim=0)
    pred_sum = torch.count_nonzero(y_pred, dim=0)
    true_sum = torch.count_nonzero(y_true, dim=0)

    fp = pred_sum - tp_sum
    fn = true_sum - tp_sum
    tp = tp_sum
    tn = y_true.shape[0] - tp - fp - fn

    return torch.stack([tn, fp, fn, tp])


y_true = torch.tensor(((1, 0), (0, 1))).cuda()
y_pred = torch.tensor(((1, 0), (0, 1))).cuda()
print(multilabel_confusion_matrix(y_true, y_pred).shape)
print(multilabel_confusion_matrix(y_true, y_pred).shape)