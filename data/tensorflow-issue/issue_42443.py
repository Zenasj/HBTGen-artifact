# tf.convert_to_tensor([B]) for inputs 'a' and 'b', where B is batch dimension (in this example, scalar inputs with batch dim)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define named inputs expected by the model
        self.input_a = tf.keras.Input(shape=(1,), name='a')
        self.input_b = tf.keras.Input(shape=(1,), name='b')
        # The model layers do a concat on axis=0, then reduce_sum on axis=0.
        # We'll implement equivalent logic in call().
        # The explicit Input objects are just for reference here;
        # model is callable with dict inputs, but does not enforce key-checking by default.
        # We'll add key validity checking logic to address the issue.
    
    def call(self, inputs):
        # inputs is expected to be a dictionary of tensors with keys possibly 'a' and 'b'
        # According to the issue, the model accepts any dict with the correct length (2),
        # regardless of key names.
        # We will add explicit key checking and raise a warning or error if keys do not match expected keys.
        
        expected_keys = ['a', 'b']
        provided_keys = list(inputs.keys())
        
        if len(provided_keys) != len(expected_keys):
            raise ValueError(
                f"Input dict length {len(provided_keys)} does not match expected {len(expected_keys)}")
        
        # Check if keys exactly match expected keys
        if set(provided_keys) != set(expected_keys):
            # Raise a warning indicating keys mismatch but tensor count matches
            tf.print(
                "Warning: Input keys do not match expected keys.",
                "Expected keys:", expected_keys,
                "Provided keys:", provided_keys)

        # Extract tensors in the order of expected keys to maintain consistent behavior
        # This mimics Keras behavior that uses nest.flatten which orders dict by keys
        # We explicitly grab values from input dict in the expected keys order.
        tensors = [inputs[k] for k in expected_keys]
        
        # Concatenate on axis=0 as per original code.
        concat = tf.concat(tensors, axis=0)
        out = tf.reduce_sum(concat, axis=0)
        return out

def my_model_function():
    return MyModel()

def GetInput():
    # Provide a dictionary with correct keys ('a' and 'b') and tensor shapes
    # Original example used scalar tensors in a batch (shape [1])
    input_a = tf.convert_to_tensor([[1.0]])  # shape (1,1) to match Input(shape=(1,))
    input_b = tf.convert_to_tensor([[1.0]])
    return {'a': input_a, 'b': input_b}

