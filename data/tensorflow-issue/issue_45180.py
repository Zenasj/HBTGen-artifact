from tensorflow.keras import layers

def _serialize_function_to_config(self, inputs, allow_raw=False):
    if isinstance(inputs, python_types.LambdaType):
      output = generic_utils.func_dump(inputs)
      output_type = 'lambda'
      module = inputs.__module__
    elif callable(inputs):
      output = inputs.__name__
      output_type = 'function'
      module = inputs.__module__
    elif allow_raw:
      output = inputs
      output_type = 'raw'
      module = None
    else:
      raise ValueError(
          'Invalid input for serialization, type: %s ' % type(inputs))

    return output, output_type, module

class SafeSaveLambda(keras.layers.Lambda):
    def _serialize_function_to_config(self, inputs, allow_raw=False):
        output, output_type, module = super()._serialize_function_to_config(inputs, allow_raw)
        
        return output, output_type, "__main__"