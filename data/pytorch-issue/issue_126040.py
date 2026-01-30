import torch

x = torch.rand(batch_size, 3, imgsz, imgsz)
dim0_x = torch.export.Dim('dim0_x', min=img_num % batch_size, max=batch_size)
dynamic_shapes = (
    {0: dim0_x,},
)
example_inputs = (x,)
exported_model = capture_pre_autograd_graph(
    model,
    example_inputs,
    dynamic_shapes=dynamic_shapes
)