import torch

def export_onnx_model(model: Net, input_data: Tuple[Any], export_name: str):
    input_names = [
        "actors",
        "actor_idcs",
        "actor_ctrs",
        "g_graph",
        "g_ctrs",
        "g_idcs",
        "g_feats",
        "g_turn",
        "g_control",
        "g_intersect",
        "g_left",
        "g_right",
    ]
    output_names = ["output1"]

    torch.onnx.export(
        model.cuda(),
        input_data,
        f"{export_name}.onnx",
        input_names=input_names,
        output_names=output_names,
        verbose=True,
        opset_version=9,
    )