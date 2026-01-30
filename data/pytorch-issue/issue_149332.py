import torch



model = torch.load("v1.pt", weights_only=False)
onnx_program = torch.onnx.export(
    model, (torch.randn([2, 3, 640, 640]),),
    input_names=['images'],
    output_names=['num_dets', 'bboxes', 'scores', 'labels'],
    dynamic_shapes=({0: torch.export.DYNAMIC},), dynamo=True

)
onnx_program.save("model.onnx")