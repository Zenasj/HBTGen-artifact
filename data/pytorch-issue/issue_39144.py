import torch.nn as nn

import torch

# Base model.
a = torch.nn.Sequential(
    torch.nn.Conv2d(7, 5, 1),
    torch.nn.Conv2d(5, 3, 1)
)

# Model with an additional layer, requires `strict=False`.
b = torch.nn.Sequential(
    torch.nn.Conv2d(7, 5, 1),
    torch.nn.Conv2d(5, 3, 1),
    torch.nn.Conv2d(3, 3, 1)
)

# Model with an additional layer and one of the layers has a different shape of weights,
# requires `strict=False` and the optional callback introduced in this PR: `tensor_check_fn`.
c = torch.nn.Sequential(
    torch.nn.Conv2d(7, 5, 1),
    torch.nn.Conv2d(3, 3, 1),
    torch.nn.Conv2d(3, 3, 1)
)

# This line works without this PR.
b.load_state_dict(a.state_dict(), strict=False)

# The next part doesn't work without this PR, because c has a layer that is of different shape.
def tensor_check_fn(param, input_param, error_msgs):
	if param.shape != input_param.shape:
		return False
	return True

c.load_state_dict(a.state_dict(), strict=False, tensor_check_fn=tensor_check_fn)

import torch
import torchvision

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model_new = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=3)

def tensor_check_fn(param, input_param, error_msgs):
	if param.shape != input_param.shape:
		return False
	return True

model_new.load_state_dict(model.state_dict(), tensor_check_fn=tensor_check_fn)

import torch
import torchvision

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

model_new = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=3)

def tensor_check_fn(key, param, input_param, error_msgs):
	if param.shape != input_param.shape:
		return False
	return True

model_new.load_state_dict(model.state_dict(), tensor_check_fn=tensor_check_fn)

def match_state_dict(
	state_dict_a: Dict[str, torch.Tensor],
	state_dict_b: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
	""" Filters state_dict_b to contain only states that are present in state_dict_a.

	Matching happens according to two criteria:
	    - Is the key present in state_dict_a?
	    - Does the state with the same key in state_dict_a have the same shape?

	Returns
	    (matched_state_dict, unmatched_state_dict)

	    States in matched_state_dict contains states from state_dict_b that are also
	    in state_dict_a and unmatched_state_dict contains states that have no
	    corresponding state in state_dict_a.

		In addition: state_dict_b = matched_state_dict U unmatched_state_dict.
	"""
	matched_state_dict = {
		key: state
		for (key, state) in state_dict_b.items()
		if key in state_dict_a and state.shape == state_dict_a[key].shape
	}
	unmatched_state_dict = {
		key: state
		for (key, state) in state_dict_b.items()
		if key not in matched_state_dict
	}
	return matched_state_dict, unmatched_state_dict