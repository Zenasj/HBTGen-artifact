def _get_trt_engine(
    onnx_model_path: str,
    input_tensors: List[InputTensor],
    build_cache: bool,
    int8: bool,
) -> bytes:
    # ...
    engine_string = builder.build_serialized_network(network, builder_config)
    # ...