import torch

@torch.no_grad()
def convert2onnx(architecture, input_size, model_file_name, device, input=None):
    if input is None:
        input = torch.ones(1, 3, input_size, input_size, dtype=torch.float).to(device)
    torch.onnx.export(architecture, input, model_file_name, export_params=True, opset_version=15, do_constant_folding=True)
    onnx_model = onnx.load(model_file_name)
    onnx.checker.check_model(onnx_model)
    opset_version = onnx_model.opset_import[0].version
    #print(onnx.helper.printable_graph(onnx_model.graph))
    return