import torch

onnx_program = torch.onnx.dynamo_export(
            TraceModel(), x, export_options=ExportOptions(op_level_debug=False)
        )

onnx_program = torch.onnx.dynamo_export(TraceModel(), x)