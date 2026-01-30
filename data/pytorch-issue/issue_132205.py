if GLOBALS.onnx_shape_inference:
        _C._jit_pass_onnx_graph_shape_type_inference(
            graph, params_dict, GLOBALS.export_onnx_opset_version
        )

if GLOBALS.onnx_shape_inference:
      try:
        _C._jit_pass_onnx_graph_shape_type_inference(
            graph, params_dict, GLOBALS.export_onnx_opset_version
        )
      except RuntimeError as exc:
         pass