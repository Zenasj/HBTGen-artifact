# torch.rand(B, C, H, W, dtype=...)  # The input shape is not explicitly provided, so we assume a generic shape (B, C, H, W) for the model input.
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Assuming a simple linear model for demonstration purposes
        self.linear = nn.Linear(10, 3)
    
    def forward(self, x):
        return self.linear(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Assuming the input shape is (B, 10) for a batch size B
    B = 2  # Example batch size
    return torch.randn(B, 10, requires_grad=True)

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
    ans = torch.gather(output, -1, argmax_values)
    ans.register_hook(lambda grad: print("Gradients:", grad))
    ans.backward(gradient=torch.ones_like(argmax_values))  # This line stops python

# Example usage
if __name__ == "__main__":
    model = my_model_function()
    input = GetInput()
    images = input.detach().clone()
    output = model(input)
    targets = torch.tensor([0, 2])
    run_grad_cam(input, images, model, output, targets, None)

