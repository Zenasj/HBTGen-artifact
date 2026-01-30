import torch.nn as nn
import torch.nn.functional as F

def run_grad_cam(input, images, model, output, targets, label_index_to_name):
    """
    Runs GradCAM on provided batch of images. The gradients and activations used for GradCAM are defined in the
    model's forward pass code. The images are saved in the root directory where the code was run from.
    :param images: Batch of images to draw the heatmap on
    :param model: The MDEQ pytorch model
    :param output: The outputs of the model for the batch
    :param targets: The ground truths for the imgs. Used to throw the heatmaps in folders.
    :param label_index_to_name: dict with label index as key and [label_code, label_name] as value.
                                Used for creating filename.
    :return: None
    """
    output = F.softmax(output, dim=1)
    argmax_values = output.max(dim=-1, keepdim=True)[1]
    model.zero_grad(set_to_none=True)
    print(output, argmax_values)
    torch.gather(output, -1, argmax_values).backward(gradient=torch.ones_like(argmax_values)) #This line stops python

model = nn.Linear(10, 3)
input = torch.randn(2, 10, requires_grad=True)
images = input.detach().clone()
output = model(input)
targets = torch.tensor([0, 2])

run_grad_cam(input, images, model, output, targets, None)

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.autograd import detect_anomaly

class MyMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args, kwargs=None):
        node = torch._C._current_autograd_node()
        print(f"Running {func} from within {node}")
        if node is not None:
            print("The Node was created at:")
            print("\n  ".join(node.metadata["traceback_"]))
        return func(*args, **kwargs or {})


with MyMode(), detect_anomaly():
    print("FW")
    a = torch.rand(10, requires_grad=True) 
    b = a.mul(2)
    b = b.div(3)
    b = b.sum()
    print("BW")
    b.backward()