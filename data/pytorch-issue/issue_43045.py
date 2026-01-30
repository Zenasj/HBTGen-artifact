import torch

def mask_rcnn_inference(pred_mask_logits, pred_instances):# Select masks corresponding to the predicted classes
    num_masks, num_classes, hmask, wmask = pred_mask_logits.shape
    class_pred = cat([i.pred_classes for i in pred_instances])
    indices = torch.arange(num_masks, device=class_pred.device)
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)
    mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
    
    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.pred_masks = prob  # (1, Hmask, Wmask)

indices = class_pred.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(num_masks, 1, hmask, wmask)
mask_probs_pred = pred_mask_logits.gather(1, indices).sigmoid()