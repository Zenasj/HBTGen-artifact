# Inputs: boxes (2, 10, 4), scores (2, 2, 10), selected_indices (10, 3)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        boxes, scores, selected_indices = inputs
        batch_inds = selected_indices[:, 0]
        cls_inds = selected_indices[:, 1]
        box_inds = selected_indices[:, 2]
        boxes_selected = boxes[batch_inds, box_inds, :]
        scores_selected = scores[batch_inds, cls_inds, box_inds]
        dets = torch.cat([boxes_selected, scores_selected[:, None]], dim=1)
        return dets, batch_inds, cls_inds

def my_model_function():
    return MyModel()

def GetInput():
    batch_size = 2
    num_box = 10
    num_class = 2
    num_det = num_box  # Derived from sample input creation
    boxes = torch.rand(batch_size, num_box, 4)
    scores = torch.rand(batch_size, num_class, num_box)
    batch_inds = torch.randint(batch_size, (num_det,), dtype=torch.long)
    cls_inds = torch.randint(num_class, (num_det,), dtype=torch.long)
    box_inds = torch.randint(num_box, (num_det,), dtype=torch.long)
    selected_indices = torch.cat([
        batch_inds[:, None],
        cls_inds[:, None],
        box_inds[:, None]
    ], dim=1)
    return (boxes, scores, selected_indices)

