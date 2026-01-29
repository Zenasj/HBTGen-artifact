# tf.random.uniform((B, 10), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    A Keras Model that preserves named dictionary inputs and outputs.
    
    This class assumes inputs and outputs passed into initialization 
    can be dicts or lists of tensors. It preserves these structures
    in self.inputs and self.outputs attributes.
    
    The call method also supports dict inputs if initialized with dict inputs.
    
    For demonstration, this model composes two submodels, showing how 
    named inputs & outputs can be preserved and used. The forward pass
    returns the dictionary outputs from the final model.
    
    Assumptions:
    - Inputs are dict of tensors keyed by string names.
    - Outputs are dict of tensors keyed by string names.
    - The "forward" signature supports dict input matching input keys.
    """
    def __init__(self, inputs, outputs, **kwargs):
        # inputs and outputs can be dict or list/tensor. We preserve datatype.
        # Internally, tf.keras.Model flattens inputs and outputs to lists,
        # but here we store originals and override `.inputs` & `.outputs`.
        super().__init__(inputs=inputs if isinstance(inputs, (list, tuple)) else list(inputs.values()),
                         outputs=outputs if isinstance(outputs, (list, tuple)) else list(outputs.values()),
                         **kwargs)

        # Save the original input and output dicts (or lists)
        self._nested_inputs = inputs
        self._nested_outputs = outputs
        
        # Map for input layers if dict
        if isinstance(inputs, dict):
            # Build dict of inputs for easy lookup in call
            self.input_dict = inputs
        else:
            self.input_dict = None

        # Store dict outputs if given
        if isinstance(outputs, dict):
            self.output_dict = outputs
        else:
            self.output_dict = None

    @property
    def inputs(self):
        # Return original type for inputs
        return self._nested_inputs

    @property
    def outputs(self):
        # Return original type for outputs
        return self._nested_outputs

    def call(self, inputs, training=None, mask=None):
        """
        Call method supports dict inputs if initialized with dict inputs.
        Outputs are returned in same structure as original outputs arg.
        """
        # If inputs was a dict at construction, expect dict input here
        if self.input_dict is not None:
            # Map dict inputs to list inputs in the order of self.input_dict values
            ordered_inputs = []
            for key in self.input_dict:
                if not isinstance(inputs, dict):
                    raise ValueError(f"Expected dict input for key '{key}', got {type(inputs)}")
                if key not in inputs:
                    raise ValueError(f"Missing input key '{key}' in inputs")
                ordered_inputs.append(inputs[key])
        else:
            ordered_inputs = inputs  # likely list or tensor

        # Compute outputs using base Model's call
        raw_outputs = super().call(ordered_inputs, training=training, mask=mask)

        # raw_outputs is list or tensor; convert to dict if original outputs was dict
        if self.output_dict is not None:
            # raw_outputs can be tensor if single output
            if not isinstance(raw_outputs, (list, tuple)):
                raw_outputs = [raw_outputs]
            # Map outputs to keys in correct order
            output_keys = list(self.output_dict.keys())
            if len(raw_outputs) != len(output_keys):
                raise ValueError("Mismatch between number of raw outputs and output dict keys")
            # Return dict of named outputs
            named_outputs = {k: v for k, v in zip(output_keys, raw_outputs)}
            return named_outputs
        
        # Otherwise, return as-is
        return raw_outputs


def my_model_function():
    """
    Creates and returns an instance of MyModel.
    
    For demonstration, builds a simple model accepting a dict input with key 'a', passing
    through Dense layers with outputs named 'b', 'c'. Then extends the model by adding
    a new Dense output 'd' connected to output 'c'. 
    
    This replicates the example from the issue conversation, preserving named inputs and outputs.
    """
    # Create base inputs (dict with key 'a')
    a = tf.keras.Input(shape=(10,), dtype=tf.float32, name='a')
    # First layer b
    b = tf.keras.layers.Dense(4, name='b')(a)
    # Second layer c
    c = tf.keras.layers.Dense(8, name='c')(b)
    
    # Create base model with named inputs and outputs dicts
    base_model = MyModel(inputs={'a': a}, outputs={'b': b, 'c': c})

    # Extend model by adding a new Dense layer 'd' on top of 'c'
    d = tf.keras.layers.Dense(16, name='d')(base_model.outputs['c'])
    
    # Create extended model with inputs preserved, outputs now include 'd'
    # Merge outputs dicts with new key 'd'
    extended_outputs = {'d': d, **base_model.outputs}
    extended_model = MyModel(inputs=base_model.inputs, outputs=extended_outputs)

    return extended_model


def GetInput():
    """
    Returns a single dict input matching the expected input of MyModel,
    i.e., a dict with key 'a' and a random float tensor of shape [batch, 10].

    Here batch size is assumed to be 1 for test purposes.
    """
    batch_size = 1
    return {'a': tf.random.uniform(shape=(batch_size, 10), dtype=tf.float32)}

