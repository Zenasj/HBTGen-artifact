import torch

torch.onnx.export(model, 
    args=points, 
    f='fsdv2.onnx',
    opset_version=13,
    input_names = ['points'], 
    output_names = ['scores_3d', 'labels_3d'],
    dynamic_axes={'points': {0: 'points_num'}},
    verbose=True)