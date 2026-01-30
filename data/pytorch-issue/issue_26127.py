import torch

def forward(self, xy):
    height, width = xy.shape[2], xy.shape[3]
    # When tracing (for onnx export this is Tensor)
    if not isinstance(height, torch.Tensor):
        height, width = torch.tensor(height), torch.tensor(width)
    i = torch.arange(width, dtype=torch.float32)
    j = torch.arange(height, dtype=torch.float32)

@torch.jit.script
def yolo_xy_dynsize_helper(height, width):
    """The input shape could be dynamic
    """
    i = torch.arange(width, dtype=torch.float32)
    j = torch.arange(height, dtype=torch.float32)
    return i, j


def forward(self, xy):
    height, width = xy.shape[2], xy.shape[3]
    # When tracing (for onnx export this is Tensor)
    if not isinstance(height, torch.Tensor):
        height, width = torch.tensor(height), torch.tensor(width)
    i, j = yolo_xy_dynsize_helper(height, width)