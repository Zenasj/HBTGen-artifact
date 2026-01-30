consts_size = sum(
      get_nbytes_of_tensor(tensor, all_cuda)
      for (name, tensor) in graph.constants.items()
      if name not in graph.folded_constants
  )

serialized_weights = b"".join(
      _to_bytes(graph.get_original_value_of_constant(name), all_cuda)
      for name in graph.constants.keys()
      if name not in graph.folded_constants
  )